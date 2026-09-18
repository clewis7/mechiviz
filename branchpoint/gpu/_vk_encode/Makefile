# Builds the Vulkan H.264 encoder that branchpoint/gpu/encode.py loads.
#
#   make                                  # needs Vulkan headers installed
#   make VULKAN_INCLUDE=/path/to/include  # or point at a Vulkan-Headers copy
#
# Headers come from libvulkan-dev on Debian/Ubuntu, vulkan-headers on Fedora,
# or a checkout of KhronosGroup/Vulkan-Headers. The library itself is loaded by
# soname, so libvulkan-dev is not needed at run time.

CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wno-missing-field-initializers
VULKAN_INCLUDE ?=

INCLUDE := $(if $(VULKAN_INCLUDE),-I$(VULKAN_INCLUDE),)

libbp_vk_encode.so: bp_vk_encode.c bp_vk_encode.h
	$(CC) $(CFLAGS) $(INCLUDE) -fPIC -shared -o $@ bp_vk_encode.c -l:libvulkan.so.1

# Encodes a synthetic NV12 pattern, with no wgpu involved, to check the
# encoder on its own: ./vkenc_test 1280x720 120 26 out.h264
vkenc_test: vkenc_test.c bp_vk_encode.c bp_vk_encode.h
	$(CC) $(CFLAGS) $(INCLUDE) -D_POSIX_C_SOURCE=200809L -o $@ \
		vkenc_test.c bp_vk_encode.c -l:libvulkan.so.1

clean:
	rm -f libbp_vk_encode.so vkenc_test

.PHONY: clean
