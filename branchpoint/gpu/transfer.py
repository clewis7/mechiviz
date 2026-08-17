"""The zero-copy step. Take wgpu buffer that tinygrad Tensor lives in, and copy it into a wgpu texture, on the device."""

from tinygrad import Tensor

# WebGPU spec constant: the required alignment of `bytes_per_row` in
# copy_buffer_to_texture.
BYTES_PER_ROW_ALIGNMENT = 256

# Texture formats; format name => bytes per texel
_BYTES_PER_TEXEL = {
    "r32float": 4,
    "rgba8unorm": 4,
    "rgba8uint": 4,
    "rg32float": 8,
    "rgba32float": 16,
    "r8unorm": 1,
}


def bytes_per_texel(fmt: str) -> int:
    """Byte width of one texel in `fmt`."""
    try:
        return _BYTES_PER_TEXEL[fmt]
    except KeyError:
        raise ValueError(
            f"unknown texture format {fmt!r}; add it to _BYTES_PER_TEXEL "
            f"(known: {sorted(_BYTES_PER_TEXEL)})"
        )


def padded_row_texels(width: int, fmt: str = "r32float") -> int:
    """Row width, in texels, once padded up to a legal `bytes_per_row`.

    Example: width=100, r32float -> 400 bytes/row, which is not a multiple of
    256, so we round up to 512 bytes -> 128 texels. The source tensor is
    allocated (H, 128) and the copy extent stays (100, H).
    """
    bpt = bytes_per_texel(fmt)
    row_bytes = width * bpt
    padded_bytes = -(-row_bytes // BYTES_PER_ROW_ALIGNMENT) * BYTES_PER_ROW_ALIGNMENT
    if padded_bytes % bpt:
        # Can only happen for texel widths that don't divide 256 (e.g. 3-byte
        # formats, which WebGPU doesn't have). Guard anyway.
        raise ValueError(f"cannot pad width {width} for format {fmt!r}")
    return padded_bytes // bpt


def needs_padding(width: int, fmt: str = "r32float") -> bool:
    """Check if `width` is not already row-aligned for `fmt`."""
    return padded_row_texels(width, fmt) != width


def buffer_handle(t):
    """Return the wgpu-py GPUBuffer backing a realized Tensor."""
    if isinstance(t, Tensor):
        # execute tensor computation graph
        t = t.realize()
        # get resulting buffer
        buf = t.uop.buffer
    else:
        buf = t
    buf.ensure_allocated()
    return buf._buf


def copy_tensor_to_texture(
    dev,
    tensor,
    texture,
    width: int,
    height: int,
    fmt: str = "r32float",
    origin=(0, 0, 0),
    mip_level: int = 0,
    synchronize: bool = True,
) -> None:
    """Copy a realized Tensor's buffer into a wgpu texture, on-GPU.

    Args:
        dev: the SharedWebGpuDevice from device.install().
        tensor: a tinygrad Tensor whose memory layout matches the texture. Its
            row stride must already be row-aligned for `fmt` — allocate it at
            (height, padded_row_texels(width, fmt), channels) if the natural
            width is not aligned.
        texture: a wgpu.GPUTexture with COPY_DST usage.
        width, height: the extent to copy, in texels. May be smaller than the
            tensor's padded row width; the pad columns are simply not copied.
        synchronize: wait for outstanding compute before submitting the copy.
            Queue ordering makes this unnecessary when the producing kernels
            were submitted on the same queue, but it is cheap insurance while
            debugging, and correct if anything ran on another queue.
    """
    tensor.realize()
    if synchronize:
        dev.synchronize()

    bpt = bytes_per_texel(fmt)

    # The tensor's own row width (in texels) determines bytes_per_row. For a
    # (H, W) scalar tensor that's W; for (H, W, 4) rgba it's still W, with the
    # channel axis folded into the texel.
    shape = tuple(tensor.shape)
    if len(shape) == 2:
        tensor_h, tensor_row_texels = shape
    elif len(shape) == 3:
        tensor_h, tensor_row_texels, _channels = shape
    else:
        raise ValueError(
            f"expected a 2D (H, W) or 3D (H, W, C) tensor, got shape {shape}"
        )

    # validate
    row_bytes = tensor_row_texels * bpt
    if row_bytes % BYTES_PER_ROW_ALIGNMENT:
        raise ValueError(
            f"row stride {row_bytes} bytes is not a multiple of "
            f"{BYTES_PER_ROW_ALIGNMENT}; allocate the source tensor with width "
            f"{padded_row_texels(width, fmt)} instead of {tensor_row_texels}"
        )
    if tensor_h < height or tensor_row_texels < width:
        raise ValueError(
            f"source tensor {tensor_h}x{tensor_row_texels} is smaller than the "
            f"requested copy extent {height}x{width}"
        )

    # encode and submit
    src = buffer_handle(tensor)
    enc = dev.wdev.create_command_encoder()
    enc.copy_buffer_to_texture(
        {
            "buffer": src,
            "offset": 0,
            "bytes_per_row": row_bytes,
            "rows_per_image": tensor_h,
        },
        {"texture": texture, "mip_level": mip_level, "origin": origin},
        (width, height, 1),
    )
    dev.wdev.queue.submit([enc.finish()])


COPY_BUFFER_ALIGNMENT = 4


def copy_tensor_to_buffer(
    dev,
    tensor,
    wgpu_buffer,
    nbytes: int | None = None,
    src_offset: int = 0,
    dst_offset: int = 0,
    synchronize: bool = True,
) -> None:
    """Copy a realized Tensor's buffer into another wgpu buffer, on-GPU.

    nbytes defaults to the whole tensor. All three of nbytes, src_offset and
    dst_offset must be multiples of 4.
    """
    tensor.realize()
    if synchronize:
        dev.synchronize()

    src = buffer_handle(tensor)
    size = src.size if nbytes is None else int(nbytes)

    for name, val in (
        ("nbytes", size),
        ("src_offset", src_offset),
        ("dst_offset", dst_offset),
    ):
        if val % COPY_BUFFER_ALIGNMENT:
            raise ValueError(
                f"{name}={val} must be a multiple of {COPY_BUFFER_ALIGNMENT}"
            )
    if src_offset + size > src.size:
        raise ValueError(
            f"source buffer is {src.size} bytes; cannot read {size} from "
            f"offset {src_offset}"
        )
    if dst_offset + size > wgpu_buffer.size:
        raise ValueError(
            f"destination buffer is {wgpu_buffer.size} bytes; cannot write "
            f"{size} at offset {dst_offset}"
        )

    enc = dev.wdev.create_command_encoder()
    enc.copy_buffer_to_buffer(src, src_offset, wgpu_buffer, dst_offset, size)
    dev.wdev.queue.submit([enc.finish()])
