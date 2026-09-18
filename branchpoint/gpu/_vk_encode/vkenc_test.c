/*
 * Exercises bp_vk_encode without wgpu. Allocates NV12 memory the same way
 * wgpu-native's exportable buffer does -- device local, dedicated, exportable
 * as an opaque fd, with the same buffer usage set -- so the encoder sees
 * exactly what it will see in the real pipeline. Frames are drawn on the host
 * into a staging buffer and copied in, standing in for the compute pass.
 *
 *   ./vkenc_test 640x480 [frames] [qp] [out.h264]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

#include "bp_vk_encode.h"

/* Must match wgpu-native's vk_usage_superset(). */
#define EXPORTER_BUFFER_USAGE                                                  \
    (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |     \
     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |  \
     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)

#define DIE(...)                                                               \
    do {                                                                       \
        fprintf(stderr, "vkenc_test: " __VA_ARGS__);                           \
        fputc('\n', stderr);                                                   \
        return 1;                                                              \
    } while (0)

#define VK(expr, what)                                                         \
    do {                                                                       \
        VkResult _r = (expr);                                                  \
        if (_r != VK_SUCCESS)                                                  \
            DIE("%s failed: %d", (what), _r);                                   \
    } while (0)

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int find_mem_type(VkPhysicalDevice phys, uint32_t bits,
                         VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

/* A white bar sweeping left to right over a luma ramp, with a chroma gradient
 * so both planes are visibly exercised. */
static void draw(uint8_t *nv12, uint32_t w, uint32_t h, uint32_t frame,
                 uint32_t frames)
{
    uint32_t bar = (uint32_t)((double)frame / frames * w);
    for (uint32_t y = 0; y < h; y++) {
        uint8_t *row = nv12 + (size_t)y * w;
        for (uint32_t x = 0; x < w; x++)
            row[x] = (x >= bar && x < bar + w / 16) ? 235
                                                    : (uint8_t)(16 + 110 * y / h);
    }
    uint8_t *uv = nv12 + (size_t)w * h;
    for (uint32_t y = 0; y < h / 2; y++) {
        uint8_t *row = uv + (size_t)y * w;
        for (uint32_t x = 0; x < w / 2; x++) {
            row[2 * x + 0] = (uint8_t)(16 + 200 * x / (w / 2));
            row[2 * x + 1] = (uint8_t)(240 - 200 * y / (h / 2));
        }
    }
}

int main(int argc, char **argv)
{
    uint32_t width = 640, height = 480, frames = 60, qp = 26;
    const char *path = "out.h264";

    if (argc > 1 && sscanf(argv[1], "%ux%u", &width, &height) != 2)
        DIE("could not parse size %s, expected WIDTHxHEIGHT", argv[1]);
    if (argc > 2)
        frames = (uint32_t)atoi(argv[2]);
    if (argc > 3)
        qp = (uint32_t)atoi(argv[3]);
    if (argc > 4)
        path = argv[4];

    uint32_t coded_w = (width + 15) / 16 * 16;
    uint32_t coded_h = (height + 15) / 16 * 16;
    VkDeviceSize nv12_size = (VkDeviceSize)coded_w * coded_h * 3 / 2;

    /* ---- stand-in for wgpu: a device that exports NV12 memory ---- */
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                              .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                 .pApplicationInfo = &app };
    VkInstance instance;
    VK(vkCreateInstance(&ici, NULL, &instance), "vkCreateInstance");

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, NULL);
    if (!n)
        DIE("no Vulkan devices");
    VkPhysicalDevice *devices = calloc(n, sizeof(*devices));
    vkEnumeratePhysicalDevices(instance, &n, devices);

    /* Pick a discrete GPU if there is one, matching what wgpu would choose. */
    VkPhysicalDevice phys = devices[0];
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            phys = devices[i];
            break;
        }
    }
    free(devices);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("source device: %s\n", props.deviceName);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof(*qf));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    int family = -1;
    for (uint32_t i = 0; i < nq && family < 0; i++)
        if (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT)
            family = (int)i;
    free(qf);
    if (family < 0)
        DIE("no transfer queue family");

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = (uint32_t)family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    const char *exts[] = { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = exts,
    };
    VkDevice device;
    VK(vkCreateDevice(phys, &dci, NULL, &device), "vkCreateDevice");
    VkQueue queue;
    vkGetDeviceQueue(device, (uint32_t)family, 0, &queue);

    /* The shared NV12 buffer, allocated the way wgpu-native does. */
    VkExternalMemoryBufferCreateInfo ext_buf = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_buf,
        .size = nv12_size,
        .usage = EXPORTER_BUFFER_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer shared;
    VK(vkCreateBuffer(device, &bci, NULL, &shared), "vkCreateBuffer (shared)");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, shared, &req);
    int type = find_mem_type(phys, req.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0)
        DIE("no device-local memory type");

    VkExportMemoryAllocateInfo export_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &export_info,
        .buffer = shared,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = req.size,
        .memoryTypeIndex = (uint32_t)type,
    };
    VkDeviceMemory shared_mem;
    VK(vkAllocateMemory(device, &mai, NULL, &shared_mem), "vkAllocateMemory (shared)");
    VK(vkBindBufferMemory(device, shared, shared_mem, 0), "vkBindBufferMemory (shared)");

    PFN_vkGetMemoryFdKHR get_fd =
        (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!get_fd)
        DIE("vkGetMemoryFdKHR unavailable");
    VkMemoryGetFdInfoKHR fd_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = shared_mem,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    int fd = -1;
    VK(get_fd(device, &fd_info, &fd), "vkGetMemoryFdKHR");
    printf("exported: fd=%d alloc_size=%llu (nv12 %llu)\n", fd,
           (unsigned long long)req.size, (unsigned long long)nv12_size);

    /* Host staging, standing in for the compute pass writing the NV12. */
    VkBufferCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = nv12_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    VkBuffer staging;
    VK(vkCreateBuffer(device, &sci, NULL, &staging), "vkCreateBuffer (staging)");
    VkMemoryRequirements sreq;
    vkGetBufferMemoryRequirements(device, staging, &sreq);
    int stype = find_mem_type(phys, sreq.memoryTypeBits,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (stype < 0)
        DIE("no host-visible memory type");
    VkMemoryAllocateInfo smai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = sreq.size,
        .memoryTypeIndex = (uint32_t)stype,
    };
    VkDeviceMemory staging_mem;
    VK(vkAllocateMemory(device, &smai, NULL, &staging_mem), "vkAllocateMemory (staging)");
    VK(vkBindBufferMemory(device, staging, staging_mem, 0), "vkBindBufferMemory (staging)");
    uint8_t *mapped = NULL;
    VK(vkMapMemory(device, staging_mem, 0, VK_WHOLE_SIZE, 0, (void **)&mapped),
       "vkMapMemory");

    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = (uint32_t)family,
    };
    VkCommandPool pool;
    VK(vkCreateCommandPool(device, &pci, NULL, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    VK(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");

    /* ------------------------------ the encoder ------------------------------ */
    char err[512] = "";
    BpEncoderInfo info = {
        .width = width,
        .height = height,
        .coded_width = coded_w,
        .coded_height = coded_h,
        .fps_num = 30,
        .fps_den = 1,
        .qp = qp,
        .vendor_id = props.vendorID,
        .device_id = props.deviceID,
        .nv12_fd = fd,
        .nv12_mem_size = req.size,
    };
    BpEncoder *enc = bp_encoder_create(&info, err, sizeof err);
    if (!enc)
        DIE("%s", err);
    printf("encoder: %s\n", bp_encoder_info(enc));

    FILE *out = fopen(path, "wb");
    if (!out)
        DIE("cannot open %s", path);

    const uint8_t *data;
    size_t size;
    bp_encoder_headers(enc, &data, &size);
    fwrite(data, 1, size, out);
    printf("headers: %zu bytes\n", size);

    size_t total = size;
    double worst = 0, sum = 0;
    for (uint32_t f = 0; f < frames; f++) {
        draw(mapped, coded_w, coded_h, f, frames);

        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &bi);
        VkBufferCopy region = { .size = nv12_size };
        vkCmdCopyBuffer(cmd, staging, shared, 1, &region);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submit = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };
        VK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
        vkQueueWaitIdle(queue);

        double t0 = now_ms();
        if (bp_encoder_encode(enc, &data, &size) < 0)
            DIE("frame %u: %s", f, bp_encoder_error(enc));
        double dt = now_ms() - t0;
        sum += dt;
        if (dt > worst)
            worst = dt;
        fwrite(data, 1, size, out);
        total += size;
        if (f < 3 || f == frames - 1)
            printf("frame %2u: %6zu bytes, %.2f ms\n", f, size, dt);
    }
    fclose(out);

    printf("%u frames, %zu bytes total (%.1f kB/frame), "
           "mean %.2f ms, worst %.2f ms -> %.0f fps\n",
           frames, total, total / 1024.0 / frames, sum / frames, worst,
           1000.0 * frames / sum);

    bp_encoder_destroy(enc);
    vkUnmapMemory(device, staging_mem);
    vkDestroyCommandPool(device, pool, NULL);
    vkDestroyBuffer(device, staging, NULL);
    vkFreeMemory(device, staging_mem, NULL);
    vkDestroyBuffer(device, shared, NULL);
    vkFreeMemory(device, shared_mem, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    printf("wrote %s\n", path);
    return 0;
}
