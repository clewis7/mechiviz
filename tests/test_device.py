"""Tests for branchpoint.gpu.device."""

import pytest

import numpy as np
from pygfx.renderers.wgpu import get_shared
from tinygrad.device import Device, Buffer
from tinygrad.helpers import DEV
from tinygrad import Tensor
from tinygrad.dtype import dtypes

from branchpoint.gpu.transfer import buffer_handle
from branchpoint.gpu import device as D
from .utils import MockGPUDevice, MockGPUBuffer

NAME = "TESTGPU"


@pytest.fixture
def clean_tinygrad():
    """Snapshot and restore every global install() touches."""

    saved_getter = Device._Device__get_canonicalized_item
    saved_opened = set(Device._opened_devices)
    saved_dev_value = DEV.value
    saved_installed = dict(D._INSTALLED)
    MockGPUDevice.reset_instances()

    yield

    Device._Device__get_canonicalized_item = saved_getter
    Device._opened_devices.clear()
    Device._opened_devices.update(saved_opened)
    DEV.value = saved_dev_value
    D._INSTALLED.clear()
    D._INSTALLED.update(saved_installed)
    MockGPUDevice.reset_instances()


@pytest.fixture
def mock_dev():
    return MockGPUDevice()


def test_install(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    assert isinstance(shared, D.SharedWebGpuDevice)
    # The backend must hold the SAME object it was handed — not a copy, not a
    # newly created device.
    assert shared.wdev is mock_dev


def test_installed_lookup(clean_tinygrad, mock_dev):
    assert D.installed(NAME) is None
    shared = D.install(mock_dev, name=NAME)
    assert D.installed(NAME) is shared


def test_install_is_idempotent(clean_tinygrad, mock_dev):
    """A second install with the same device must reuse the first backend."""
    first = D.install(mock_dev, name=NAME)
    second = D.install(mock_dev, name=NAME)
    assert first is second


def test_install_rejects_same_name(clean_tinygrad, mock_dev):
    D.install(mock_dev, name=NAME)
    other = MockGPUDevice(label="second")
    with pytest.raises(RuntimeError, match="already installed"):
        D.install(other, name=NAME)


def test_tinygrad_device_lookup(clean_tinygrad, mock_dev):
    """Single device for tinygrad and renderer."""
    shared = D.install(mock_dev, name=NAME)
    assert Device[NAME] is shared
    # Indexed forms must resolve too — tinygrad canonicalizes "NAME:0".
    assert Device[f"{NAME}:0"] is shared


def test_set_default_tinygrad(clean_tinygrad, mock_dev):
    """Assert device allocation with tinygrad works."""
    D.install(mock_dev, name=NAME, set_default=True)
    assert Device.DEFAULT == NAME


def test_single_device(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    for _ in range(20):
        shared.allocator._alloc(1024, None)
    assert MockGPUDevice.instance_count() == 1


def test_check_allocation_owner(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    bufs = [shared.allocator._alloc(n, None) for n in (16, 64, 256, 1000)]

    assert len(mock_dev.buffers) == 4
    for b in bufs:
        assert b.owner is mock_dev, "buffer was created on a foreign device"
        assert b in mock_dev.buffers


def test_check_buffer_copy_property(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    buf = shared.allocator._alloc(256, None)

    assert buf.usage & D.BUF_COPY_SRC, "buffers must be usable as a copy source"
    assert buf.usage & D.BUF_COPY_DST, "buffers must be writable from the host"
    assert buf.usage & D.BUF_STORAGE, "buffers must be readable by compute shaders"


def test_allocation_alignment(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    for requested, expected in [(1, 4), (3, 4), (4, 4), (5, 8), (255, 256), (257, 260)]:
        buf = shared.allocator._alloc(requested, None)
        assert buf.size == expected, (
            f"alloc({requested}) -> {buf.size}, want {expected}"
        )


def test_copyin_copyout_round_trip(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    src = np.arange(64, dtype=np.float32)
    buf = shared.allocator._alloc(src.nbytes, None)

    shared.allocator._copyin(buf, memoryview(src).cast("B"))
    out = np.empty(64, dtype=np.float32)
    shared.allocator._copyout(memoryview(out).cast("B"), buf)

    np.testing.assert_array_equal(src, out)


def test_copyin_pads_unaligned_lengths(clean_tinygrad, mock_dev):
    """A 3-byte write must not fail; wgpu wants 4-byte-aligned lengths."""
    shared = D.install(mock_dev, name=NAME)
    buf = shared.allocator._alloc(3, None)
    shared.allocator._copyin(buf, memoryview(bytearray(b"abc")))
    assert bytes(buf._data[:3]) == b"abc"


def test_read_buffer_uses_a_staging_buffer_and_cleans_up(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    buf = shared.allocator._alloc(16, None)
    mock_dev.queue.write_buffer(buf, 0, b"0123456789abcdef")

    before = len(mock_dev.buffers)
    data = shared.read_buffer(buf)

    assert data == b"0123456789abcdef"
    assert len(mock_dev.buffers) == before + 1, "expected one staging buffer"
    staging = mock_dev.buffers[-1]
    assert staging.usage & D.BUF_MAP_READ
    assert staging.destroyed, "staging buffer should be released"


def test_free_destroys_the_buffer(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    buf = shared.allocator._alloc(32, None)
    shared.allocator._free(buf, None)
    assert buf.destroyed


def test_synchronize_polls_the_device(clean_tinygrad, mock_dev):
    shared = D.install(mock_dev, name=NAME)
    shared.synchronize()
    assert mock_dev.polls == 1


def test_tinygrad_buffer_handle_is_the_wgpu_buffer(clean_tinygrad, mock_dev):
    """Tensor -> .uop -> .buffer -> ._buf must land on OUR wgpu object."""
    D.install(mock_dev, name=NAME)
    b = Buffer(NAME, 64, dtypes.float)
    b.ensure_allocated()

    handle = buffer_handle(b)
    assert isinstance(handle, MockGPUBuffer)
    assert handle.owner is mock_dev
    assert handle is b._buf, "buffer_handle must return the allocator's object itself"


def test_program_dispatch_stays_on_the_shared_device(clean_tinygrad, mock_dev):
    """A compiled kernel must build its pipeline and bind group on our device."""
    shared = D.install(mock_dev, name=NAME)
    prog = D.SharedWebGPUProgram(shared, "my_kernel", b"@compute fn my_kernel() {}")
    buf = shared.allocator._alloc(64, None)

    prog(buf, global_size=(4, 1, 1), vals=(7,))

    assert mock_dev.shader_modules, "shader module was not created on our device"
    assert ("dispatch", (4, 1, 1)) in mock_dev.dispatches
    assert mock_dev.queue.submissions >= 1


def _has_real_gpu() -> bool:
    try:
        get_shared().device
        return True
    except Exception:
        return False


real_gpu = pytest.mark.skipif(not _has_real_gpu(), reason="needs a GPU and pygfx")


@real_gpu
def test_real_device_is_pygfx_device(clean_tinygrad):
    shared = D.install(name="WEBGPU")
    assert shared.wdev is get_shared().device


@real_gpu
def test_real_tensor_math_and_buffer_identity(clean_tinygrad):
    """End to end: a tensor computed on the shared device, whose buffer is a
    wgpu object created by pygfx's device."""
    shared = D.install(name="WEBGPU")
    t = (Tensor(np.arange(16, dtype=np.float32)) * 2).realize()

    np.testing.assert_allclose(t.numpy(), np.arange(16) * 2)
    handle = buffer_handle(t)
    assert type(handle).__name__.startswith("GPUBuffer")
