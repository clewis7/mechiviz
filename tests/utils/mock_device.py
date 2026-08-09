"""A mock wgpu GPUDevice to test device sharing without a GPU."""


class MockGPUBuffer:
    """Stands in for wgpu.GPUBuffer. Actually stores bytes so copies are real."""

    def __init__(self, owner: "MockGPUDevice", size: int, usage: int, label: str = ""):
        self.owner = owner  # which device created it — the key assertion
        self.size = size
        self.usage = usage
        self.label = label
        self.destroyed = False
        self._data = bytearray(size)
        self._mapped = False

    # -- host mapping (used by SharedWebGpuDevice.read_buffer) --
    def map_sync(self, mode):
        self._mapped = True

    def read_mapped(self):
        if not self._mapped:
            raise RuntimeError("read_mapped() on an unmapped buffer")
        return bytes(self._data)

    def unmap(self):
        self._mapped = False

    def destroy(self):
        self.destroyed = True

    def __repr__(self):
        return f"<MockGPUBuffer size={self.size} usage=0x{self.usage:x}>"


class MockGPUTexture:
    def __init__(self, owner, width, height, fmt):
        self.owner = owner
        self.width, self.height, self.format = width, height, fmt
        self.writes = []  # every copy that landed here


class MockQueue:
    def __init__(self, owner):
        self.owner = owner
        self.submissions = 0

    def write_buffer(self, buf, offset, data):
        if buf.owner is not self.owner:
            raise RuntimeError("write_buffer on a buffer from a different device!")
        mv = memoryview(data).cast("B")
        buf._data[offset : offset + len(mv)] = bytes(mv)

    def submit(self, command_buffers):
        self.submissions += 1


class MockComputePass:
    def __init__(self, enc):
        self.enc = enc

    def set_pipeline(self, p):
        self.enc.device.dispatches.append(("pipeline", p))

    def set_bind_group(self, index, group):
        pass

    def dispatch_workgroups(self, x, y=1, z=1):
        self.enc.device.dispatches.append(("dispatch", (x, y, z)))

    def end(self):
        pass


class MockCommandEncoder:
    """Executes copies immediately; finish() returns an opaque token."""

    def __init__(self, device):
        self.device = device

    def begin_compute_pass(self):
        return MockComputePass(self)

    def copy_buffer_to_buffer(self, src, src_off, dst, dst_off, size):
        for b in (src, dst):
            if b.owner is not self.device:
                raise RuntimeError("copy across devices! this is the bug we test for")
        dst._data[dst_off : dst_off + size] = src._data[src_off : src_off + size]

    def copy_buffer_to_texture(self, source, destination, copy_size):
        buf = source["buffer"]
        if buf.owner is not self.device:
            raise RuntimeError("copy_buffer_to_texture across devices!")
        bpr = source["bytes_per_row"]
        # Mirror the constraint the real API enforces, so tests catch violations.
        if bpr % 256:
            raise ValueError(f"bytes_per_row={bpr} is not a multiple of 256")
        record = {
            "bytes_per_row": bpr,
            "rows_per_image": source.get("rows_per_image"),
            "copy_size": tuple(copy_size),
            "origin": destination.get("origin"),
            "buffer": buf,
        }
        self.device.texture_copies.append(record)
        tex = destination["texture"]
        if isinstance(tex, MockGPUTexture):
            tex.writes.append(record)

    def finish(self):
        return object()


class MockGPUDevice:
    """Stands in for wgpu.GPUDevice, recording everything worth asserting on."""

    _instances: list["MockGPUDevice"] = []

    def __init__(self, label: str = "mock"):
        self.label = label
        self.buffers: list[MockGPUBuffer] = []
        self.textures: list[MockGPUTexture] = []
        self.shader_modules: list[str] = []
        self.pipelines: list[object] = []
        self.dispatches: list[tuple] = []
        self.texture_copies: list[dict] = []
        self.polls = 0
        self.queue = MockQueue(self)
        MockGPUDevice._instances.append(self)

    # -- the assertion helper the tests care most about --
    @classmethod
    def instance_count(cls) -> int:
        """How many mock devices have ever been constructed this session."""
        return len(cls._instances)

    @classmethod
    def reset_instances(cls):
        cls._instances = []

    # -- resource creation --
    def create_buffer(self, size, usage, label="", **kw):
        b = MockGPUBuffer(self, size, int(usage), label)
        self.buffers.append(b)
        return b

    def create_texture(self, size, format, usage, dimension="2d", **kw):
        t = MockGPUTexture(self, size[0], size[1], format)
        self.textures.append(t)
        return t

    def create_shader_module(self, code, **kw):
        self.shader_modules.append(code)
        return ("module", len(self.shader_modules) - 1)

    def create_bind_group_layout(self, entries, **kw):
        return ("bgl", tuple(e["binding"] for e in entries))

    def create_pipeline_layout(self, bind_group_layouts, **kw):
        return ("pl", tuple(bind_group_layouts))

    def create_compute_pipeline(self, layout, compute, **kw):
        p = ("pipeline", compute.get("entry_point"))
        self.pipelines.append(p)
        return p

    def create_bind_group(self, layout, entries, **kw):
        for e in entries:
            buf = e["resource"].get("buffer")
            if buf is not None and buf.owner is not self:
                raise RuntimeError("bind group references a foreign device's buffer!")
        return ("bg", len(entries))

    def create_command_encoder(self, **kw):
        return MockCommandEncoder(self)

    def _poll(self):
        self.polls += 1

    def __repr__(self):
        return f"<MockGPUDevice {self.label!r} buffers={len(self.buffers)}>"
