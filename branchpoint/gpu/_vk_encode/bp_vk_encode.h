/* All-intra H.264 encoder on Vulkan Video. See bp_vk_encode.c. */

#ifndef BP_VK_ENCODE_H
#define BP_VK_ENCODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct BpEncoder BpEncoder;

typedef struct BpEncoderInfo {
    uint32_t width;         /* displayed picture size */
    uint32_t height;
    uint32_t coded_width;   /* macroblock-aligned size of the NV12 source */
    uint32_t coded_height;
    uint32_t fps_num;       /* used for the SPS timing info only */
    uint32_t fps_den;
    uint32_t qp;            /* 0..51, constant for every slice */
    uint32_t vendor_id;     /* PCI ids of the device that exported the memory, */
    uint32_t device_id;     /* so the encoder picks that same physical device */
    int nv12_fd;            /* OPAQUE_FD for the NV12 source memory */
    uint64_t nv12_mem_size; /* size of the exporter's allocation */
} BpEncoderInfo;

/* Creates an encoder for the given picture size. On success the file
 * descriptor is owned by the encoder and must not be closed by the caller; on
 * failure it is left untouched and a message is written to `err`. */
BpEncoder *bp_encoder_create(const BpEncoderInfo *info, char *err, size_t err_size);

void bp_encoder_destroy(BpEncoder *enc);

/* SPS and PPS as Annex B NAL units, valid for the encoder's lifetime. */
void bp_encoder_headers(const BpEncoder *enc, const uint8_t **data, size_t *size);

/* Encodes whatever the NV12 source currently holds as one IDR access unit.
 * `*data` points into host-visible memory and stays valid until the next call.
 * Returns 0, or -1 with a message available from bp_encoder_error(). */
int bp_encoder_encode(BpEncoder *enc, const uint8_t **data, size_t *size);

const char *bp_encoder_error(const BpEncoder *enc);

/* One line naming the device and the settings actually in use. */
const char *bp_encoder_info(const BpEncoder *enc);

#endif
