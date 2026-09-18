/*
 * All-intra H.264 encoder built on Vulkan Video.
 *
 * The encoder owns its own VkInstance/VkDevice, because wgpu will neither
 * enable the video extensions nor hand out a video-encode queue. The only
 * resource shared with the renderer is one block of device memory holding the
 * NV12 source picture, imported here from an OPAQUE_FD handle. Per frame:
 *
 *     imported NV12 buffer --copy--> NV12 image --encode--> bitstream buffer
 *                                                           (host visible)
 *
 * Every picture is an IDR, so there are no reference lists to manage: the
 * session is reset each frame, and the reconstructed picture goes into DPB
 * slot 0 and is never read back.
 */

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vulkan/vulkan.h>

#include "bp_vk_encode.h"

#define ERR_SIZE 512
#define INFO_SIZE 512

/* Table A-1 of the H.264 specification. Bitrate limits are omitted; the
 * encoder runs at constant QP so there is no rate to check. Level 1b has no
 * distinct level_idc and is skipped. */
static const struct {
    StdVideoH264LevelIdc idc;
    uint32_t max_mbps;    /* macroblocks per second */
    uint32_t max_fs;      /* macroblocks per frame */
    uint32_t max_dpb_mbs;
} H264_LEVELS[] = {
    { STD_VIDEO_H264_LEVEL_IDC_1_0,     1485,     99,    396 },
    { STD_VIDEO_H264_LEVEL_IDC_1_1,     3000,    396,    900 },
    { STD_VIDEO_H264_LEVEL_IDC_1_2,     6000,    396,   2376 },
    { STD_VIDEO_H264_LEVEL_IDC_1_3,    11880,    396,   2376 },
    { STD_VIDEO_H264_LEVEL_IDC_2_0,    11880,    396,   2376 },
    { STD_VIDEO_H264_LEVEL_IDC_2_1,    19800,    792,   4752 },
    { STD_VIDEO_H264_LEVEL_IDC_2_2,    20250,   1620,   8100 },
    { STD_VIDEO_H264_LEVEL_IDC_3_0,    40500,   1620,   8100 },
    { STD_VIDEO_H264_LEVEL_IDC_3_1,   108000,   3600,  18000 },
    { STD_VIDEO_H264_LEVEL_IDC_3_2,   216000,   5120,  20480 },
    { STD_VIDEO_H264_LEVEL_IDC_4_0,   245760,   8192,  32768 },
    { STD_VIDEO_H264_LEVEL_IDC_4_1,   245760,   8192,  32768 },
    { STD_VIDEO_H264_LEVEL_IDC_4_2,   522240,   8704,  34816 },
    { STD_VIDEO_H264_LEVEL_IDC_5_0,   589824,  22080, 110400 },
    { STD_VIDEO_H264_LEVEL_IDC_5_1,   983040,  36864, 184320 },
    { STD_VIDEO_H264_LEVEL_IDC_5_2,  2073600,  36864, 184320 },
    { STD_VIDEO_H264_LEVEL_IDC_6_0,  4177920, 139264, 696320 },
    { STD_VIDEO_H264_LEVEL_IDC_6_1,  8355840, 139264, 696320 },
    { STD_VIDEO_H264_LEVEL_IDC_6_2, 16711680, 139264, 696320 },
};

struct BpEncoder {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    uint32_t queue_family;
    VkQueue queue;

    uint32_t width, height, coded_w, coded_h;
    uint32_t qp;
    uint32_t vendor_id, device_id;

    /* One profile description, referenced from every resource we create. */
    VkVideoEncodeH264ProfileInfoKHR h264_profile;
    VkVideoProfileInfoKHR profile;
    VkVideoProfileListInfoKHR profile_list;
    VkVideoCapabilitiesKHR caps;
    VkVideoEncodeCapabilitiesKHR enc_caps;
    VkVideoEncodeH264CapabilitiesKHR h264_caps;

    VkVideoSessionKHR session;
    VkDeviceMemory *session_mem;
    uint32_t n_session_mem;
    VkVideoSessionParametersKHR params;

    VkBuffer src_buffer;          /* imported: NV12 written by the renderer */
    VkDeviceMemory src_mem;

    VkImage in_image;             /* VIDEO_ENCODE_SRC, NV12 */
    VkDeviceMemory in_mem;
    VkImageView in_view;

    VkImage dpb_image;            /* VIDEO_ENCODE_DPB, one slot */
    VkDeviceMemory dpb_mem;
    VkImageView dpb_view;

    VkBuffer bs_buffer;           /* VIDEO_ENCODE_DST, host visible */
    VkDeviceMemory bs_mem;
    VkDeviceSize bs_size;
    uint8_t *bs_mapped;
    int bs_coherent;

    VkCommandPool pool;
    VkCommandBuffer cmd;
    VkFence fence;
    VkQueryPool query_pool;

    uint8_t *headers;
    size_t headers_size;
    int has_overrides;

    uint32_t frame_index;
    int cabac;

    /* Video entry points, all of which must come from vkGetDeviceProcAddr. */
    PFN_vkCreateVideoSessionKHR CreateVideoSessionKHR;
    PFN_vkDestroyVideoSessionKHR DestroyVideoSessionKHR;
    PFN_vkGetVideoSessionMemoryRequirementsKHR GetVideoSessionMemoryRequirementsKHR;
    PFN_vkBindVideoSessionMemoryKHR BindVideoSessionMemoryKHR;
    PFN_vkCreateVideoSessionParametersKHR CreateVideoSessionParametersKHR;
    PFN_vkDestroyVideoSessionParametersKHR DestroyVideoSessionParametersKHR;
    PFN_vkGetEncodedVideoSessionParametersKHR GetEncodedVideoSessionParametersKHR;
    PFN_vkCmdBeginVideoCodingKHR CmdBeginVideoCodingKHR;
    PFN_vkCmdEndVideoCodingKHR CmdEndVideoCodingKHR;
    PFN_vkCmdControlVideoCodingKHR CmdControlVideoCodingKHR;
    PFN_vkCmdEncodeVideoKHR CmdEncodeVideoKHR;

    char err[ERR_SIZE];
    char info[INFO_SIZE];
};

/* ------------------------------------------------------------------ helpers */

static int fail(char *err, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, ERR_SIZE, fmt, ap);
    va_end(ap);
    return -1;
}

static const char *res_str(VkResult r)
{
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR:
        return "VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR:
        return "VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR:
        return "VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR:
        return "VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR";
    case VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR:
        return "VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR";
    default: return "VkResult";
    }
}

#define CHECK(expr, what)                                                      \
    do {                                                                       \
        VkResult _r = (expr);                                                  \
        if (_r != VK_SUCCESS)                                                  \
            return fail(err, "%s: %s (%d)", (what), res_str(_r), _r);          \
    } while (0)

static int find_mem_type(VkPhysicalDevice phys, uint32_t type_bits,
                         VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((type_bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & want) == want)
            return (int)i;
    return -1;
}

static int has_ext(const VkExtensionProperties *ep, uint32_t n, const char *name)
{
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(ep[i].extensionName, name))
            return 1;
    return 0;
}

/* Lowest level whose picture-size and rate limits cover this stream. */
static int pick_level(uint32_t mb_w, uint32_t mb_h, uint32_t fps,
                      StdVideoH264LevelIdc max_idc, StdVideoH264LevelIdc *out,
                      char *err)
{
    uint32_t fs = mb_w * mb_h;
    for (size_t i = 0; i < sizeof(H264_LEVELS) / sizeof(*H264_LEVELS); i++) {
        if (H264_LEVELS[i].idc > max_idc)
            break;
        if (fs > H264_LEVELS[i].max_fs || fs > H264_LEVELS[i].max_dpb_mbs ||
            fs * fps > H264_LEVELS[i].max_mbps)
            continue;
        /* A.3.1: neither dimension may exceed sqrt(MaxFS * 8) macroblocks. */
        if (mb_w * mb_w > H264_LEVELS[i].max_fs * 8 ||
            mb_h * mb_h > H264_LEVELS[i].max_fs * 8)
            continue;
        *out = H264_LEVELS[i].idc;
        return 0;
    }
    return fail(err, "%ux%u macroblocks at %u fps exceeds every H.264 level the "
                     "encoder supports (max level_idc %u)",
                mb_w, mb_h, fps, max_idc);
}

/* ------------------------------------------------------------- device set-up */

static const char *const DEVICE_EXTENSIONS[] = {
    VK_KHR_VIDEO_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME,
    VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
};
#define N_DEVICE_EXTENSIONS (sizeof(DEVICE_EXTENSIONS) / sizeof(*DEVICE_EXTENSIONS))

static int create_instance(BpEncoder *enc, char *err)
{
    const char *layer = "VK_LAYER_KHRONOS_validation";
    const char *layers[1];
    uint32_t n_layers = 0;

    if (getenv("BP_VK_VALIDATION")) {
        uint32_t n = 0;
        vkEnumerateInstanceLayerProperties(&n, NULL);
        VkLayerProperties *lp = calloc(n, sizeof(*lp));
        if (!lp)
            return fail(err, "out of memory");
        vkEnumerateInstanceLayerProperties(&n, lp);
        for (uint32_t i = 0; i < n; i++) {
            if (!strcmp(lp[i].layerName, layer)) {
                layers[0] = layer;
                n_layers = 1;
                break;
            }
        }
        free(lp);
        if (!n_layers)
            fprintf(stderr, "bp_vk_encode: %s requested but not installed\n", layer);
    }

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "branchpoint",
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
        .enabledLayerCount = n_layers,
        .ppEnabledLayerNames = n_layers ? layers : NULL,
    };
    CHECK(vkCreateInstance(&ci, NULL, &enc->instance), "vkCreateInstance");
    return 0;
}

/* Fills in the profile/capability structs; the profile pointers must stay
 * valid for as long as the encoder lives, so they live in `enc`. */
static void init_profile(BpEncoder *enc)
{
    enc->h264_profile = (VkVideoEncodeH264ProfileInfoKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR,
        .stdProfileIdc = STD_VIDEO_H264_PROFILE_IDC_MAIN,
    };
    enc->profile = (VkVideoProfileInfoKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        .pNext = &enc->h264_profile,
        .videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
        .chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        .lumaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        .chromaBitDepth = VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };
    enc->profile_list = (VkVideoProfileListInfoKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
        .profileCount = 1,
        .pProfiles = &enc->profile,
    };
    enc->h264_caps = (VkVideoEncodeH264CapabilitiesKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR,
    };
    enc->enc_caps = (VkVideoEncodeCapabilitiesKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR,
        .pNext = &enc->h264_caps,
    };
    enc->caps = (VkVideoCapabilitiesKHR){
        .sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        .pNext = &enc->enc_caps,
    };
}

/* The physical device that exported the NV12 memory, which must also be able
 * to encode H.264 Main 4:2:0 8-bit from a queue family that can copy. Matching
 * on the PCI ids matters: an imported handle is only valid on the device that
 * exported it, and other drivers in the loader's list may also offer H.264
 * encode. With no ids given, the first capable device wins. */
/* Whether this device can encode the profile, and from which queue family.
 * `caps` receives the capabilities, `reason` a message when it cannot. */
static int device_can_encode(VkInstance instance, VkPhysicalDevice phys,
                             const VkVideoProfileInfoKHR *profile,
                             VkVideoCapabilitiesKHR *caps, uint32_t *queue_family,
                             char *reason, size_t reason_size)
{
    PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR get_caps =
        (PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR)vkGetInstanceProcAddr(
            instance, "vkGetPhysicalDeviceVideoCapabilitiesKHR");

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);

    if (props.apiVersion < VK_API_VERSION_1_3) {
        snprintf(reason, reason_size, "%s reports Vulkan %u.%u, need 1.3",
                 props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
                 VK_API_VERSION_MINOR(props.apiVersion));
        return 0;
    }

    uint32_t n_ext = 0;
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n_ext, NULL);
    VkExtensionProperties *ext = calloc(n_ext ? n_ext : 1, sizeof(*ext));
    if (!ext) {
        snprintf(reason, reason_size, "out of memory");
        return 0;
    }
    vkEnumerateDeviceExtensionProperties(phys, NULL, &n_ext, ext);
    const char *missing = NULL;
    for (size_t k = 0; k < N_DEVICE_EXTENSIONS; k++)
        if (!has_ext(ext, n_ext, DEVICE_EXTENSIONS[k]))
            missing = DEVICE_EXTENSIONS[k];
    free(ext);
    if (missing) {
        snprintf(reason, reason_size, "%s does not support %s", props.deviceName,
                 missing);
        return 0;
    }

    uint32_t n_qf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n_qf, NULL);
    VkQueueFamilyProperties *qf = calloc(n_qf ? n_qf : 1, sizeof(*qf));
    if (!qf) {
        snprintf(reason, reason_size, "out of memory");
        return 0;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &n_qf, qf);
    int family = -1;
    for (uint32_t q = 0; q < n_qf; q++) {
        VkQueueFlags need = VK_QUEUE_VIDEO_ENCODE_BIT_KHR | VK_QUEUE_TRANSFER_BIT;
        if ((qf[q].queueFlags & need) == need) {
            family = (int)q;
            break;
        }
    }
    free(qf);
    if (family < 0) {
        snprintf(reason, reason_size, "%s has no queue family supporting both "
                 "video encode and transfer operations", props.deviceName);
        return 0;
    }

    VkResult r = get_caps(phys, profile, caps);
    if (r != VK_SUCCESS) {
        snprintf(reason, reason_size,
                 "%s cannot encode H.264 Main 4:2:0 8-bit: %s", props.deviceName,
                 res_str(r));
        return 0;
    }

    *queue_family = (uint32_t)family;
    return 1;
}

static int pick_device(BpEncoder *enc, char *err)
{
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(enc->instance, &n, NULL),
          "vkEnumeratePhysicalDevices");
    VkPhysicalDevice *devices = calloc(n ? n : 1, sizeof(*devices));
    if (!devices)
        return fail(err, "out of memory");
    vkEnumeratePhysicalDevices(enc->instance, &n, devices);

    char why[ERR_SIZE] = "no Vulkan devices";
    size_t seen = 0;
    char listing[ERR_SIZE] = "";
    int found = 0;

    for (uint32_t i = 0; i < n && !found; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        seen += snprintf(listing + seen, sizeof listing - seen,
                         "%s[%u] %s (0x%04x:0x%04x)", seen ? ", " : "", i,
                         props.deviceName, props.vendorID, props.deviceID);
        if (seen >= sizeof listing)
            seen = sizeof listing - 1;

        if ((enc->vendor_id || enc->device_id) &&
            (props.vendorID != enc->vendor_id || props.deviceID != enc->device_id))
            continue;

        if (!device_can_encode(enc->instance, devices[i], &enc->profile,
                               &enc->caps, &enc->queue_family, why, sizeof why))
            continue;

        enc->phys = devices[i];
        snprintf(enc->info, INFO_SIZE, "%s (0x%04x:0x%04x)", props.deviceName,
                 props.vendorID, props.deviceID);
        found = 1;
    }

    free(devices);
    if (!found) {
        if (enc->vendor_id || enc->device_id)
            return fail(err, "no H.264 encode device matching 0x%04x:0x%04x, the "
                             "device that exported the NV12 memory; Vulkan "
                             "offers %s",
                        enc->vendor_id, enc->device_id, listing);
        return fail(err, "no usable H.264 encode device: %s", why);
    }
    return 0;
}

static int create_device(BpEncoder *enc, char *err)
{
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = enc->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkPhysicalDeviceVulkan13Features features13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = N_DEVICE_EXTENSIONS,
        .ppEnabledExtensionNames = DEVICE_EXTENSIONS,
    };
    CHECK(vkCreateDevice(enc->phys, &dci, NULL, &enc->device), "vkCreateDevice");
    vkGetDeviceQueue(enc->device, enc->queue_family, 0, &enc->queue);

#define LOAD(name)                                                             \
    do {                                                                       \
        enc->name = (PFN_vk##name)vkGetDeviceProcAddr(enc->device, "vk" #name); \
        if (!enc->name)                                                        \
            return fail(err, "vkGetDeviceProcAddr(vk" #name ") returned NULL"); \
    } while (0)

    LOAD(CreateVideoSessionKHR);
    LOAD(DestroyVideoSessionKHR);
    LOAD(GetVideoSessionMemoryRequirementsKHR);
    LOAD(BindVideoSessionMemoryKHR);
    LOAD(CreateVideoSessionParametersKHR);
    LOAD(DestroyVideoSessionParametersKHR);
    LOAD(GetEncodedVideoSessionParametersKHR);
    LOAD(CmdBeginVideoCodingKHR);
    LOAD(CmdEndVideoCodingKHR);
    LOAD(CmdControlVideoCodingKHR);
    LOAD(CmdEncodeVideoKHR);
#undef LOAD
    return 0;
}

/* --------------------------------------------------------------- resources */

/* The two-plane NV12 format is the only one this encoder handles; every
 * driver with H.264 encode support reports it. */
static int pick_format(BpEncoder *enc, VkImageUsageFlags usage,
                       VkImageUsageFlags also_needed, VkImageTiling *tiling,
                       char *err)
{
    PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR get_fmt =
        (PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR)vkGetInstanceProcAddr(
            enc->instance, "vkGetPhysicalDeviceVideoFormatPropertiesKHR");

    VkPhysicalDeviceVideoFormatInfoKHR info = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR,
        .pNext = &enc->profile_list,
        .imageUsage = usage,
    };
    uint32_t n = 0;
    CHECK(get_fmt(enc->phys, &info, &n, NULL),
          "vkGetPhysicalDeviceVideoFormatPropertiesKHR");
    if (!n)
        return fail(err, "driver reports no image formats for usage 0x%x", usage);

    VkVideoFormatPropertiesKHR *props = calloc(n, sizeof(*props));
    if (!props)
        return fail(err, "out of memory");
    for (uint32_t i = 0; i < n; i++)
        props[i].sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
    VkResult r = get_fmt(enc->phys, &info, &n, props);
    if (r != VK_SUCCESS) {
        free(props);
        return fail(err, "vkGetPhysicalDeviceVideoFormatPropertiesKHR: %s",
                    res_str(r));
    }

    int ok = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (props[i].format != VK_FORMAT_G8_B8R8_2PLANE_420_UNORM)
            continue;
        if ((props[i].imageUsageFlags & (usage | also_needed)) !=
            (usage | also_needed))
            continue;
        *tiling = props[i].imageTiling;
        ok = 1;
        break;
    }
    free(props);
    if (!ok)
        return fail(err, "driver does not support G8_B8R8_2PLANE_420_UNORM with "
                         "usage 0x%x", usage | also_needed);
    return 0;
}

static int create_video_image(BpEncoder *enc, VkImageTiling tiling,
                              VkImageUsageFlags usage, VkImage *image,
                              VkDeviceMemory *mem, VkImageView *view, char *err)
{
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &enc->profile_list,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .extent = { enc->coded_w, enc->coded_h, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = tiling,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    CHECK(vkCreateImage(enc->device, &ici, NULL, image), "vkCreateImage");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(enc->device, *image, &req);
    int type = find_mem_type(enc->phys, req.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0)
        type = find_mem_type(enc->phys, req.memoryTypeBits, 0);
    if (type < 0)
        return fail(err, "no memory type for a video image");

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = (uint32_t)type,
    };
    CHECK(vkAllocateMemory(enc->device, &mai, NULL, mem),
          "vkAllocateMemory (video image)");
    CHECK(vkBindImageMemory(enc->device, *image, *mem, 0), "vkBindImageMemory");

    VkImageViewCreateInfo vci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    CHECK(vkCreateImageView(enc->device, &vci, NULL, view), "vkCreateImageView");
    return 0;
}

static int create_session(BpEncoder *enc, char *err)
{
    VkExtensionProperties std_header = {
        .specVersion = VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION,
    };
    snprintf(std_header.extensionName, sizeof std_header.extensionName, "%s",
             VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME);

    VkVideoSessionCreateInfoKHR sci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
        .queueFamilyIndex = enc->queue_family,
        .pVideoProfile = &enc->profile,
        .pictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .maxCodedExtent = { enc->coded_w, enc->coded_h },
        .referencePictureFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
        .maxDpbSlots = 1,
        .maxActiveReferencePictures = 1,
        .pStdHeaderVersion = &std_header,
    };
    CHECK(enc->CreateVideoSessionKHR(enc->device, &sci, NULL, &enc->session),
          "vkCreateVideoSessionKHR");

    uint32_t n = 0;
    CHECK(enc->GetVideoSessionMemoryRequirementsKHR(enc->device, enc->session,
                                                    &n, NULL),
          "vkGetVideoSessionMemoryRequirementsKHR");
    if (!n)
        return 0;

    VkVideoSessionMemoryRequirementsKHR *req = calloc(n, sizeof(*req));
    VkBindVideoSessionMemoryInfoKHR *bind = calloc(n, sizeof(*bind));
    enc->session_mem = calloc(n, sizeof(*enc->session_mem));
    if (!req || !bind || !enc->session_mem) {
        free(req);
        free(bind);
        return fail(err, "out of memory");
    }
    enc->n_session_mem = n;
    for (uint32_t i = 0; i < n; i++)
        req[i].sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;

    VkResult r = enc->GetVideoSessionMemoryRequirementsKHR(enc->device,
                                                           enc->session, &n, req);
    if (r != VK_SUCCESS) {
        free(req);
        free(bind);
        return fail(err, "vkGetVideoSessionMemoryRequirementsKHR: %s", res_str(r));
    }

    for (uint32_t i = 0; i < n; i++) {
        int type = find_mem_type(enc->phys, req[i].memoryRequirements.memoryTypeBits,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type < 0)
            type = find_mem_type(enc->phys,
                                 req[i].memoryRequirements.memoryTypeBits, 0);
        if (type < 0) {
            free(req);
            free(bind);
            return fail(err, "no memory type for video session bind %u", i);
        }
        VkMemoryAllocateInfo mai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = req[i].memoryRequirements.size,
            .memoryTypeIndex = (uint32_t)type,
        };
        r = vkAllocateMemory(enc->device, &mai, NULL, &enc->session_mem[i]);
        if (r != VK_SUCCESS) {
            free(req);
            free(bind);
            return fail(err, "vkAllocateMemory (session): %s", res_str(r));
        }
        bind[i] = (VkBindVideoSessionMemoryInfoKHR){
            .sType = VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
            .memoryBindIndex = req[i].memoryBindIndex,
            .memory = enc->session_mem[i],
            .memoryOffset = 0,
            .memorySize = req[i].memoryRequirements.size,
        };
    }
    r = enc->BindVideoSessionMemoryKHR(enc->device, enc->session, n, bind);
    free(req);
    free(bind);
    CHECK(r, "vkBindVideoSessionMemoryKHR");
    return 0;
}

/* Builds the SPS/PPS, hands them to the driver, and reads back the encoded
 * parameter sets: those bytes are what the stream must start with, including
 * any syntax the driver overrode. */
static int create_params(BpEncoder *enc, uint32_t fps_num, uint32_t fps_den,
                         char *err)
{
    StdVideoH264LevelIdc level;
    uint32_t fps = fps_den ? (fps_num + fps_den - 1) / fps_den : 30;
    if (pick_level(enc->coded_w / 16, enc->coded_h / 16, fps,
                   enc->h264_caps.maxLevelIdc, &level, err) < 0)
        return -1;

    enc->cabac = (enc->h264_caps.stdSyntaxFlags &
                  VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR) != 0;

    StdVideoH264SequenceParameterSetVui vui = {
        .flags = {
            .video_signal_type_present_flag = 1,
            .video_full_range_flag = 0,
            .color_description_present_flag = 1,
            .timing_info_present_flag = 1,
            .fixed_frame_rate_flag = 1,
        },
        .aspect_ratio_idc = STD_VIDEO_H264_ASPECT_RATIO_IDC_UNSPECIFIED,
        .video_format = 5,             /* unspecified, table E-2 */
        .colour_primaries = 1,         /* BT.709 */
        .transfer_characteristics = 1, /* BT.709 */
        .matrix_coefficients = 1,      /* BT.709 */
        .num_units_in_tick = fps_den ? fps_den : 1,
        .time_scale = 2 * (fps_num ? fps_num : 30),
    };
    StdVideoH264SequenceParameterSet sps = {
        .flags = {
            .constraint_set1_flag = 1,
            .constraint_set4_flag = 1,  /* frame_mbs_only_flag is 1 */
            .constraint_set5_flag = 1,  /* no B slices */
            .direct_8x8_inference_flag = 1,
            .frame_mbs_only_flag = 1,
            .frame_cropping_flag = (enc->coded_w != enc->width ||
                                    enc->coded_h != enc->height),
            .vui_parameters_present_flag = 1,
        },
        .profile_idc = STD_VIDEO_H264_PROFILE_IDC_MAIN,
        .level_idc = level,
        .chroma_format_idc = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        .pic_order_cnt_type = STD_VIDEO_H264_POC_TYPE_2,
        .max_num_ref_frames = 1,
        .pic_width_in_mbs_minus1 = enc->coded_w / 16 - 1,
        .pic_height_in_map_units_minus1 = enc->coded_h / 16 - 1,
        /* Crop offsets count chroma samples for 4:2:0 progressive. */
        .frame_crop_right_offset = (enc->coded_w - enc->width) / 2,
        .frame_crop_bottom_offset = (enc->coded_h - enc->height) / 2,
        .pSequenceParameterSetVui = &vui,
    };
    StdVideoH264PictureParameterSet pps = {
        .flags = {
            .entropy_coding_mode_flag = enc->cabac,
        },
        .weighted_bipred_idc = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
    };

    VkVideoEncodeH264SessionParametersAddInfoKHR add = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
        .stdSPSCount = 1,
        .pStdSPSs = &sps,
        .stdPPSCount = 1,
        .pStdPPSs = &pps,
    };
    VkVideoEncodeH264SessionParametersCreateInfoKHR h264_params = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .maxStdSPSCount = 1,
        .maxStdPPSCount = 1,
        .pParametersAddInfo = &add,
    };
    VkVideoEncodeQualityLevelInfoKHR quality = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
        .pNext = &h264_params,
        .qualityLevel = 0,
    };
    VkVideoSessionParametersCreateInfoKHR pci = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = &quality,
        .videoSession = enc->session,
    };
    CHECK(enc->CreateVideoSessionParametersKHR(enc->device, &pci, NULL,
                                               &enc->params),
          "vkCreateVideoSessionParametersKHR");

    VkVideoEncodeH264SessionParametersGetInfoKHR h264_get = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR,
        .writeStdSPS = VK_TRUE,
        .writeStdPPS = VK_TRUE,
        .stdSPSId = 0,
        .stdPPSId = 0,
    };
    VkVideoEncodeSessionParametersGetInfoKHR get = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
        .pNext = &h264_get,
        .videoSessionParameters = enc->params,
    };
    VkVideoEncodeH264SessionParametersFeedbackInfoKHR h264_feedback = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
    };
    VkVideoEncodeSessionParametersFeedbackInfoKHR feedback = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_FEEDBACK_INFO_KHR,
        .pNext = &h264_feedback,
    };

    size_t size = 0;
    VkResult r = enc->GetEncodedVideoSessionParametersKHR(enc->device, &get,
                                                          &feedback, &size, NULL);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !size)
        return fail(err, "vkGetEncodedVideoSessionParametersKHR sizing: %s (%zu "
                         "bytes)", res_str(r), size);
    enc->headers = malloc(size);
    if (!enc->headers)
        return fail(err, "out of memory");
    CHECK(enc->GetEncodedVideoSessionParametersKHR(enc->device, &get, &feedback,
                                                   &size, enc->headers),
          "vkGetEncodedVideoSessionParametersKHR");
    enc->headers_size = size;
    enc->has_overrides = feedback.hasOverrides;

    size_t len = strlen(enc->info);
    snprintf(enc->info + len, INFO_SIZE - len,
             ", H.264 Main level_idc %u, %ux%u (coded %ux%u), all-intra, "
             "%s, QP %u%s",
             level, enc->width, enc->height, enc->coded_w, enc->coded_h,
             enc->cabac ? "CABAC" : "CAVLC", enc->qp,
             enc->has_overrides ? ", driver overrode SPS/PPS syntax" : "");
    return 0;
}

/* The exporting side allocates dedicated memory, so the import must be
 * dedicated too, bound to an identically created buffer: same size, same
 * usage, same sharing mode. wgpu-native's exportable buffer always uses the
 * usage superset below whatever usage the caller asked wgpu for, so this
 * mirrors it rather than asking only for what the copy needs.
 *
 * The file descriptor is consumed by a successful import. */
#define BP_EXPORTER_BUFFER_USAGE                                               \
    (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |     \
     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |  \
     VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)

static int import_source(BpEncoder *enc, int fd, uint64_t mem_size, char *err)
{
    VkDeviceSize nv12_size = (VkDeviceSize)enc->coded_w * enc->coded_h * 3 / 2;

    VkExternalMemoryBufferCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext,
        .size = nv12_size,
        .usage = BP_EXPORTER_BUFFER_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    CHECK(vkCreateBuffer(enc->device, &bci, NULL, &enc->src_buffer),
          "vkCreateBuffer (NV12 source)");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(enc->device, enc->src_buffer, &req);
    /* A dedicated import must be the same size as the export, and both come
     * from an identically created buffer, so these should be equal. */
    if (req.size != mem_size)
        return fail(err, "exported allocation is %llu bytes but an identically "
                         "created %llu byte NV12 buffer needs %llu; the "
                         "exporter and importer disagree",
                    (unsigned long long)mem_size,
                    (unsigned long long)nv12_size,
                    (unsigned long long)req.size);

    /* An opaque handle carries no memory type of its own -- querying one with
     * vkGetMemoryFdPropertiesKHR is forbidden for OPAQUE_FD. Because the
     * buffer here is created identically to the exporter's, on the same
     * physical device, picking the first device-local type out of its
     * requirements reproduces the exporter's choice. */
    int type = find_mem_type(enc->phys, req.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type < 0)
        return fail(err, "no device-local memory type accepts the NV12 buffer "
                         "(memoryTypeBits 0x%x)", req.memoryTypeBits);

    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = enc->src_buffer,
    };
    VkImportMemoryFdInfoKHR import = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = &dedicated,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        .fd = fd,
    };
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import,
        .allocationSize = mem_size,
        .memoryTypeIndex = (uint32_t)type,
    };
    CHECK(vkAllocateMemory(enc->device, &mai, NULL, &enc->src_mem),
          "vkAllocateMemory (import NV12 handle)");
    CHECK(vkBindBufferMemory(enc->device, enc->src_buffer, enc->src_mem, 0),
          "vkBindBufferMemory (NV12 source)");
    return 0;
}

static int create_bitstream(BpEncoder *enc, char *err)
{
    /* An intra frame at a low QP can approach the size of the raw picture;
     * twice that plus a margin for the headers is comfortable. */
    VkDeviceSize size = (VkDeviceSize)enc->coded_w * enc->coded_h * 3 + (1 << 16);
    VkDeviceSize align = enc->caps.minBitstreamBufferSizeAlignment;
    if (align)
        size = (size + align - 1) / align * align;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &enc->profile_list,
        .size = size,
        .usage = VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    CHECK(vkCreateBuffer(enc->device, &bci, NULL, &enc->bs_buffer),
          "vkCreateBuffer (bitstream)");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(enc->device, enc->bs_buffer, &req);

    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    int type = find_mem_type(enc->phys, req.memoryTypeBits, want);
    if (type < 0) {
        want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        type = find_mem_type(enc->phys, req.memoryTypeBits, want);
    }
    if (type < 0) {
        want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        type = find_mem_type(enc->phys, req.memoryTypeBits, want);
    }
    if (type < 0)
        return fail(err, "no host-visible memory type for the bitstream buffer");
    enc->bs_coherent = (want & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = (uint32_t)type,
    };
    CHECK(vkAllocateMemory(enc->device, &mai, NULL, &enc->bs_mem),
          "vkAllocateMemory (bitstream)");
    CHECK(vkBindBufferMemory(enc->device, enc->bs_buffer, enc->bs_mem, 0),
          "vkBindBufferMemory (bitstream)");
    CHECK(vkMapMemory(enc->device, enc->bs_mem, 0, VK_WHOLE_SIZE, 0,
                      (void **)&enc->bs_mapped),
          "vkMapMemory (bitstream)");
    enc->bs_size = size;
    return 0;
}

static int create_commands(BpEncoder *enc, char *err)
{
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = enc->queue_family,
    };
    CHECK(vkCreateCommandPool(enc->device, &pci, NULL, &enc->pool),
          "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = enc->pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    CHECK(vkAllocateCommandBuffers(enc->device, &cai, &enc->cmd),
          "vkAllocateCommandBuffers");

    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    CHECK(vkCreateFence(enc->device, &fci, NULL, &enc->fence), "vkCreateFence");

    VkVideoEncodeFeedbackFlagsKHR feedback =
        VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR |
        VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR;
    if ((enc->enc_caps.supportedEncodeFeedbackFlags & feedback) != feedback)
        return fail(err, "driver does not report bitstream offset/size encode "
                         "feedback (0x%x)",
                    enc->enc_caps.supportedEncodeFeedbackFlags);

    VkQueryPoolVideoEncodeFeedbackCreateInfoKHR fci2 = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
        .pNext = &enc->profile,
        .encodeFeedbackFlags = feedback,
    };
    VkQueryPoolCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .pNext = &fci2,
        .queryType = VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
        .queryCount = 1,
    };
    CHECK(vkCreateQueryPool(enc->device, &qci, NULL, &enc->query_pool),
          "vkCreateQueryPool");
    return 0;
}

/* The reconstructed picture must be in the DPB layout before first use. Its
 * contents never matter, since nothing ever references it. */
static int transition_dpb(BpEncoder *enc, char *err)
{
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK(vkBeginCommandBuffer(enc->cmd, &bi), "vkBeginCommandBuffer (dpb)");

    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_WRITE_BIT_KHR,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = enc->dpb_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(enc->cmd, &dep);
    CHECK(vkEndCommandBuffer(enc->cmd), "vkEndCommandBuffer (dpb)");

    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &enc->cmd,
    };
    CHECK(vkQueueSubmit(enc->queue, 1, &si, enc->fence), "vkQueueSubmit (dpb)");
    CHECK(vkWaitForFences(enc->device, 1, &enc->fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences (dpb)");
    CHECK(vkResetFences(enc->device, 1, &enc->fence), "vkResetFences (dpb)");
    return 0;
}

/* -------------------------------------------------------------- public API */

static int check_limits(BpEncoder *enc, char *err)
{
    if (enc->coded_w % 16 || enc->coded_h % 16)
        return fail(err, "coded size %ux%u must be a multiple of 16",
                    enc->coded_w, enc->coded_h);
    if (enc->coded_w < enc->width || enc->coded_h < enc->height)
        return fail(err, "coded size %ux%u is smaller than the picture %ux%u",
                    enc->coded_w, enc->coded_h, enc->width, enc->height);

    uint32_t gw = enc->enc_caps.encodeInputPictureGranularity.width;
    uint32_t gh = enc->enc_caps.encodeInputPictureGranularity.height;
    if ((gw && enc->coded_w % gw) || (gh && enc->coded_h % gh))
        return fail(err, "coded size %ux%u is not a multiple of the encoder's "
                         "input granularity %ux%u",
                    enc->coded_w, enc->coded_h, gw, gh);

    if (enc->coded_w < enc->caps.minCodedExtent.width ||
        enc->coded_h < enc->caps.minCodedExtent.height ||
        enc->coded_w > enc->caps.maxCodedExtent.width ||
        enc->coded_h > enc->caps.maxCodedExtent.height)
        return fail(err, "coded size %ux%u is outside the encoder's range "
                         "%ux%u..%ux%u",
                    enc->coded_w, enc->coded_h,
                    enc->caps.minCodedExtent.width, enc->caps.minCodedExtent.height,
                    enc->caps.maxCodedExtent.width, enc->caps.maxCodedExtent.height);

    if (!(enc->enc_caps.rateControlModes &
          VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR))
        return fail(err, "encoder cannot disable rate control, so a constant QP "
                         "is unavailable (modes 0x%x)",
                    enc->enc_caps.rateControlModes);

    if ((int32_t)enc->qp < enc->h264_caps.minQp ||
        (int32_t)enc->qp > enc->h264_caps.maxQp)
        return fail(err, "QP %u is outside the encoder's range [%d, %d]",
                    enc->qp, enc->h264_caps.minQp, enc->h264_caps.maxQp);

    if (enc->caps.maxDpbSlots < 1 || enc->caps.maxActiveReferencePictures < 1)
        return fail(err, "encoder reports maxDpbSlots %u / "
                         "maxActiveReferencePictures %u, need at least 1 of each",
                    enc->caps.maxDpbSlots, enc->caps.maxActiveReferencePictures);
    return 0;
}

BpEncoder *bp_encoder_create(const BpEncoderInfo *info, char *err, size_t err_size)
{
    char local[ERR_SIZE] = "";
    BpEncoder *enc = calloc(1, sizeof(*enc));
    if (!enc) {
        snprintf(err, err_size, "out of memory");
        return NULL;
    }
    enc->width = info->width;
    enc->height = info->height;
    enc->coded_w = info->coded_width;
    enc->coded_h = info->coded_height;
    enc->qp = info->qp;
    enc->vendor_id = info->vendor_id;
    enc->device_id = info->device_id;

    VkImageTiling src_tiling = VK_IMAGE_TILING_OPTIMAL;
    VkImageTiling dpb_tiling = VK_IMAGE_TILING_OPTIMAL;
    int imported = 0;

    if (create_instance(enc, local) < 0)
        goto fail;
    init_profile(enc);
    if (pick_device(enc, local) < 0)
        goto fail;
    if (create_device(enc, local) < 0)
        goto fail;
    if (check_limits(enc, local) < 0)
        goto fail;
    if (pick_format(enc, VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR,
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT, &src_tiling, local) < 0)
        goto fail;
    if (pick_format(enc, VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR, 0,
                    &dpb_tiling, local) < 0)
        goto fail;
    if (create_session(enc, local) < 0)
        goto fail;
    if (create_params(enc, info->fps_num, info->fps_den, local) < 0)
        goto fail;
    if (create_video_image(enc, src_tiling,
                           VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                           &enc->in_image, &enc->in_mem, &enc->in_view, local) < 0)
        goto fail;
    if (create_video_image(enc, dpb_tiling,
                           VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR,
                           &enc->dpb_image, &enc->dpb_mem, &enc->dpb_view,
                           local) < 0)
        goto fail;
    if (import_source(enc, info->nv12_fd, info->nv12_mem_size, local) < 0)
        goto fail;
    imported = 1;
    if (create_bitstream(enc, local) < 0)
        goto fail;
    if (create_commands(enc, local) < 0)
        goto fail;
    if (transition_dpb(enc, local) < 0)
        goto fail;
    return enc;

fail:
    snprintf(err, err_size, "%s", local);
    if (!imported)
        enc->src_mem = VK_NULL_HANDLE; /* the handle was never consumed */
    bp_encoder_destroy(enc);
    return NULL;
}

int bp_encoder_encode(BpEncoder *enc, const uint8_t **data, size_t *size)
{
    char *err = enc->err;

    CHECK(vkResetCommandBuffer(enc->cmd, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    CHECK(vkBeginCommandBuffer(enc->cmd, &bi), "vkBeginCommandBuffer");

    vkCmdResetQueryPool(enc->cmd, enc->query_pool, 0, 1);

    /* The renderer's writes to the shared memory are ordered before this
     * submission by the host-side wait the caller does first; the barrier
     * only makes them visible to the copy. The input image is overwritten in
     * full, so its old contents can be discarded. */
    VkBufferMemoryBarrier2 src_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = enc->src_buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    VkImageMemoryBarrier2 to_transfer = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = enc->in_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier2(enc->cmd, &(VkDependencyInfo){
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &src_barrier,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_transfer,
    });

    /* bufferRowLength and imageExtent are in the texels of the selected
     * plane: plane 0 is R8 at full resolution, plane 1 is R8G8 at half. */
    VkBufferImageCopy2 regions[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset = 0,
            .bufferRowLength = enc->coded_w,
            .bufferImageHeight = enc->coded_h,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT,
                .layerCount = 1,
            },
            .imageExtent = { enc->coded_w, enc->coded_h, 1 },
        },
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferOffset = (VkDeviceSize)enc->coded_w * enc->coded_h,
            .bufferRowLength = enc->coded_w / 2,
            .bufferImageHeight = enc->coded_h / 2,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT,
                .layerCount = 1,
            },
            .imageExtent = { enc->coded_w / 2, enc->coded_h / 2, 1 },
        },
    };
    vkCmdCopyBufferToImage2(enc->cmd, &(VkCopyBufferToImageInfo2){
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer = enc->src_buffer,
        .dstImage = enc->in_image,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount = 2,
        .pRegions = regions,
    });

    VkImageMemoryBarrier2 to_encode = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_VIDEO_ENCODE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_VIDEO_ENCODE_READ_BIT_KHR,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = enc->in_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier2(enc->cmd, &(VkDependencyInfo){
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &to_encode,
    });

    VkVideoPictureResourceInfoKHR src_res = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        .codedExtent = { enc->width, enc->height },
        .imageViewBinding = enc->in_view,
    };
    VkVideoPictureResourceInfoKHR dpb_res = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        .codedExtent = { enc->width, enc->height },
        .imageViewBinding = enc->dpb_view,
    };

    StdVideoEncodeH264ReferenceInfo ref_info = {
        .primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_IDR,
    };
    VkVideoEncodeH264DpbSlotInfoKHR dpb_slot_info = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR,
        .pStdReferenceInfo = &ref_info,
    };
    /* The session is reset every frame, so slot 0 always starts out inactive:
     * the resource is declared unassociated (-1) and then activated by being
     * named as this picture's reconstructed output. */
    VkVideoReferenceSlotInfoKHR declared_slot = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .pNext = &dpb_slot_info,
        .slotIndex = -1,
        .pPictureResource = &dpb_res,
    };
    VkVideoReferenceSlotInfoKHR setup_slot = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
        .pNext = &dpb_slot_info,
        .slotIndex = 0,
        .pPictureResource = &dpb_res,
    };

    StdVideoEncodeH264SliceHeader slice_header = {
        .first_mb_in_slice = 0,
        .slice_type = STD_VIDEO_H264_SLICE_TYPE_I,
        /* pic_init_qp is left at its default of 26. */
        .slice_qp_delta = (int8_t)((int32_t)enc->qp - 26),
        .cabac_init_idc = STD_VIDEO_H264_CABAC_INIT_IDC_0,
        .disable_deblocking_filter_idc =
            STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
    };
    VkVideoEncodeH264NaluSliceInfoKHR slice = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
        .constantQp = (int32_t)enc->qp,
        .pStdSliceHeader = &slice_header,
    };

    /* An I slice has no reference lists; the entries are still filled with
     * "no reference picture" so nothing reads uninitialised indices. */
    StdVideoEncodeH264ReferenceListsInfo ref_lists = { 0 };
    memset(ref_lists.RefPicList0, STD_VIDEO_H264_NO_REFERENCE_PICTURE,
           sizeof ref_lists.RefPicList0);
    memset(ref_lists.RefPicList1, STD_VIDEO_H264_NO_REFERENCE_PICTURE,
           sizeof ref_lists.RefPicList1);

    StdVideoEncodeH264PictureInfo pic_info = {
        .flags = { .IdrPicFlag = 1, .is_reference = 1 },
        /* Consecutive IDR pictures must not share an idr_pic_id (7.4.3). */
        .idr_pic_id = (uint16_t)(enc->frame_index & 1),
        .primary_pic_type = STD_VIDEO_H264_PICTURE_TYPE_IDR,
        .frame_num = 0,
        .PicOrderCnt = 0,
        .pRefLists = &ref_lists,
    };
    VkVideoEncodeH264GopRemainingFrameInfoKHR gop = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_GOP_REMAINING_FRAME_INFO_KHR,
        .useGopRemainingFrames = VK_TRUE,
        .gopRemainingI = 1,
    };
    VkVideoEncodeH264PictureInfoKHR h264_pic = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
        .pNext = &gop,
        .naluSliceEntryCount = 1,
        .pNaluSliceEntries = &slice,
        .pStdPictureInfo = &pic_info,
    };
    VkVideoEncodeInfoKHR encode_info = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
        .pNext = &h264_pic,
        .dstBuffer = enc->bs_buffer,
        .dstBufferOffset = 0,
        .dstBufferRange = enc->bs_size,
        .srcPictureResource = src_res,
        .pSetupReferenceSlot = &setup_slot,
    };

    VkVideoEncodeRateControlInfoKHR rate_control = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
        .rateControlMode = VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR,
    };
    VkVideoEncodeQualityLevelInfoKHR quality = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_QUALITY_LEVEL_INFO_KHR,
        .pNext = &rate_control,
        .qualityLevel = 0,
    };
    VkVideoCodingControlInfoKHR control = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
        .pNext = &quality,
        .flags = VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR |
                 VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR |
                 VK_VIDEO_CODING_CONTROL_ENCODE_QUALITY_LEVEL_BIT_KHR,
    };
    VkVideoBeginCodingInfoKHR begin = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
        .videoSession = enc->session,
        .videoSessionParameters = enc->params,
        .referenceSlotCount = 1,
        .pReferenceSlots = &declared_slot,
    };
    VkVideoEndCodingInfoKHR end = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    };

    enc->CmdBeginVideoCodingKHR(enc->cmd, &begin);
    enc->CmdControlVideoCodingKHR(enc->cmd, &control);
    vkCmdBeginQuery(enc->cmd, enc->query_pool, 0, 0);
    enc->CmdEncodeVideoKHR(enc->cmd, &encode_info);
    vkCmdEndQuery(enc->cmd, enc->query_pool, 0);
    enc->CmdEndVideoCodingKHR(enc->cmd, &end);

    CHECK(vkEndCommandBuffer(enc->cmd), "vkEndCommandBuffer");
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &enc->cmd,
    };
    CHECK(vkQueueSubmit(enc->queue, 1, &submit, enc->fence), "vkQueueSubmit");
    CHECK(vkWaitForFences(enc->device, 1, &enc->fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    CHECK(vkResetFences(enc->device, 1, &enc->fence), "vkResetFences");

    /* Feedback values come back in bit order: offset, then bytes written,
     * then the status appended by VK_QUERY_RESULT_WITH_STATUS_BIT_KHR. */
    uint32_t feedback[3] = { 0, 0, 0 };
    CHECK(vkGetQueryPoolResults(enc->device, enc->query_pool, 0, 1,
                                sizeof feedback, feedback, sizeof feedback,
                                VK_QUERY_RESULT_WAIT_BIT |
                                VK_QUERY_RESULT_WITH_STATUS_BIT_KHR),
          "vkGetQueryPoolResults");
    if ((int32_t)feedback[2] != VK_QUERY_RESULT_STATUS_COMPLETE_KHR)
        return fail(err, "encode did not complete: status %d",
                    (int32_t)feedback[2]);

    if (!enc->bs_coherent) {
        VkMappedMemoryRange range = {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = enc->bs_mem,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        CHECK(vkInvalidateMappedMemoryRanges(enc->device, 1, &range),
              "vkInvalidateMappedMemoryRanges");
    }

    *data = enc->bs_mapped + feedback[0];
    *size = feedback[1];
    enc->frame_index++;
    return 0;
}

void bp_encoder_headers(const BpEncoder *enc, const uint8_t **data, size_t *size)
{
    *data = enc->headers;
    *size = enc->headers_size;
}

const char *bp_encoder_error(const BpEncoder *enc)
{
    return enc->err;
}

const char *bp_encoder_info(const BpEncoder *enc)
{
    return enc->info;
}

void bp_encoder_destroy(BpEncoder *enc)
{
    if (!enc)
        return;
    if (enc->device) {
        vkDeviceWaitIdle(enc->device);
        if (enc->query_pool)
            vkDestroyQueryPool(enc->device, enc->query_pool, NULL);
        if (enc->fence)
            vkDestroyFence(enc->device, enc->fence, NULL);
        if (enc->pool)
            vkDestroyCommandPool(enc->device, enc->pool, NULL);
        if (enc->bs_mapped)
            vkUnmapMemory(enc->device, enc->bs_mem);
        if (enc->bs_buffer)
            vkDestroyBuffer(enc->device, enc->bs_buffer, NULL);
        if (enc->bs_mem)
            vkFreeMemory(enc->device, enc->bs_mem, NULL);
        if (enc->src_buffer)
            vkDestroyBuffer(enc->device, enc->src_buffer, NULL);
        if (enc->src_mem)
            vkFreeMemory(enc->device, enc->src_mem, NULL);
        if (enc->dpb_view)
            vkDestroyImageView(enc->device, enc->dpb_view, NULL);
        if (enc->dpb_image)
            vkDestroyImage(enc->device, enc->dpb_image, NULL);
        if (enc->dpb_mem)
            vkFreeMemory(enc->device, enc->dpb_mem, NULL);
        if (enc->in_view)
            vkDestroyImageView(enc->device, enc->in_view, NULL);
        if (enc->in_image)
            vkDestroyImage(enc->device, enc->in_image, NULL);
        if (enc->in_mem)
            vkFreeMemory(enc->device, enc->in_mem, NULL);
        if (enc->params)
            enc->DestroyVideoSessionParametersKHR(enc->device, enc->params, NULL);
        if (enc->session)
            enc->DestroyVideoSessionKHR(enc->device, enc->session, NULL);
        for (uint32_t i = 0; i < enc->n_session_mem; i++)
            if (enc->session_mem[i])
                vkFreeMemory(enc->device, enc->session_mem[i], NULL);
        vkDestroyDevice(enc->device, NULL);
    }
    if (enc->instance)
        vkDestroyInstance(enc->instance, NULL);
    free(enc->session_mem);
    free(enc->headers);
    free(enc);
}
