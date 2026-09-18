"""Linear RGBA texture -> NV12 in a storage buffer, one compute pass.

NV12 layout written into the buffer:
    offset 0                   Y  plane, coded_h rows of coded_w bytes
    offset coded_w * coded_h   CbCr plane, coded_h/2 rows of coded_w bytes
                               (Cb and Cr interleaved)

One invocation covers a 4x2 pixel block: two u32 of luma (four Y samples each)
and one u32 of chroma (two Cb/Cr pairs), so every store is a whole word.
"""

import wgpu


SHADER = """
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> nv12: array<u32>;

const WORDS_PER_ROW: u32 = {words_per_row}u;   // coded_w / 4
const CHROMA_ROWS: u32 = {chroma_rows}u;       // coded_h / 2
const UV_WORD_BASE: u32 = {uv_word_base}u;     // coded_w * coded_h / 4
const SRC_MAX: vec2<i32> = vec2<i32>({src_max_x}, {src_max_y});

fn load(x: i32, y: i32) -> vec3<f32> {{
    let c = textureLoad(src, min(vec2<i32>(x, y), SRC_MAX), 0).rgb;
    // A pygfx render target hands back linear light either way: a plain
    // format stores linear values, and an -srgb view linearises on load.
    // BT.709 Y'CbCr is defined on gamma-encoded R'G'B', so encode here.
    let lo = c * 12.92;
    let hi = pow(c, vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}}

// BT.709 limited range, on gamma-encoded R'G'B'.
fn luma(c: vec3<f32>) -> f32 {{
    return dot(c, vec3<f32>(46.559, 156.629, 15.812)) + 16.0;
}}

fn chroma(c: vec3<f32>) -> vec2<f32> {{
    return vec2<f32>(dot(c, vec3<f32>(-25.664, -86.336, 112.0)),
                     dot(c, vec3<f32>(112.0, -101.730, -10.270))) + 128.0;
}}

fn pack(a: f32, b: f32, c: f32, d: f32) -> u32 {{
    let v = clamp(round(vec4<f32>(a, b, c, d)), vec4<f32>(0.0), vec4<f32>(255.0));
    return pack4x8unorm(v / 255.0);
}}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= WORDS_PER_ROW || gid.y >= CHROMA_ROWS) {{
        return;
    }}
    let x = i32(gid.x) * 4;
    let y = i32(gid.y) * 2;

    let p00 = load(x, y);
    let p10 = load(x + 1, y);
    let p20 = load(x + 2, y);
    let p30 = load(x + 3, y);
    let p01 = load(x, y + 1);
    let p11 = load(x + 1, y + 1);
    let p21 = load(x + 2, y + 1);
    let p31 = load(x + 3, y + 1);

    let row = gid.y * 2u * WORDS_PER_ROW + gid.x;
    nv12[row] = pack(luma(p00), luma(p10), luma(p20), luma(p30));
    nv12[row + WORDS_PER_ROW] = pack(luma(p01), luma(p11), luma(p21), luma(p31));

    // Chroma is box-filtered over each 2x2 block. The transform is affine, so
    // averaging R'G'B' first is identical to averaging Cb/Cr afterwards.
    let left = chroma((p00 + p10 + p01 + p11) * 0.25);
    let right = chroma((p20 + p30 + p21 + p31) * 0.25);
    nv12[UV_WORD_BASE + gid.y * WORDS_PER_ROW + gid.x] =
        pack(left.x, left.y, right.x, right.y);
}}
"""

def _check_format(fmt: str) -> None:
    """Reject formats the shader cannot read as linear rgba.

    bgra8unorm describes byte order in memory only, so textureLoad hands back
    components in rgba order for every supported format.
    """
    if fmt.removesuffix("-srgb") not in ("rgba8unorm", "bgra8unorm", "rgba16float"):
        raise ValueError(
            f"unsupported texture format {fmt!r}; expected rgba8unorm, "
            f"bgra8unorm or rgba16float, optionally -srgb"
        )


class Nv12Converter:
    """Compute pipeline that writes `texture` into `buffer` as NV12.

    Parameters
    ----------
    device : wgpu.GPUDevice
    texture : wgpu.GPUTexture
        Must have TEXTURE_BINDING usage.
    coded_size : (int, int)
        (width, height) of the NV12 output, each a multiple of 16. May exceed
        the texture size; the extra columns/rows repeat the edge texel.
    buffer : wgpu.GPUBuffer
        At least `coded_w * coded_h * 3 // 2` bytes, STORAGE usage.
    """

    def __init__(self, device, texture, coded_size, buffer):
        coded_w, coded_h = coded_size
        if coded_w % 16 or coded_h % 16:
            raise ValueError(f"coded size {coded_w}x{coded_h} must be a multiple of 16")
        nbytes = coded_w * coded_h * 3 // 2
        if buffer.size < nbytes:
            raise ValueError(f"buffer is {buffer.size} bytes, need {nbytes}")

        self.device = device
        self.coded_size = (coded_w, coded_h)
        self.nbytes = nbytes
        self._groups = (-(-(coded_w // 4) // 8), -(-(coded_h // 2) // 8), 1)

        _check_format(texture.format)
        code = SHADER.format(
            words_per_row=coded_w // 4,
            chroma_rows=coded_h // 2,
            uv_word_base=coded_w * coded_h // 4,
            src_max_x=texture.size[0] - 1,
            src_max_y=texture.size[1] - 1,
        )
        layout = device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": wgpu.BufferBindingType.storage},
                },
            ]
        )
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={
                "module": device.create_shader_module(code=code),
                "entry_point": "main",
            },
        )
        self._bind_group = device.create_bind_group(
            layout=layout,
            entries=[
                {"binding": 0, "resource": texture.create_view()},
                {"binding": 1, "resource": {"buffer": buffer, "offset": 0, "size": nbytes}},
            ],
        )

    def record(self, encoder) -> None:
        """Record the conversion into an open command encoder."""
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self._pipeline)
        cpass.set_bind_group(0, self._bind_group)
        cpass.dispatch_workgroups(*self._groups)
        cpass.end()
