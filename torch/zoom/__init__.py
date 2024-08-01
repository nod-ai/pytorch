import importlib
import os
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, Callable, cast, List, Optional, Tuple, Union

import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import _dummy_type, _LazySeedTracker, classproperty

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: List[
    Tuple[Callable[[], None], List[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = getattr(torch._C, "_zoom_isInBadFork", lambda: False)
_device_t = Union[_device, str, int, None]
_lazy_seed_tracker = _LazySeedTracker()
_cached_device_count: Optional[int] = None

class DeferredZoomCallError(Exception):
    pass

def get_amp_supported_dtype() -> List[torch.dtype]:
    return [torch.float16, torch.bfloat16, torch.float32]

def _is_compiled() -> bool:
    r"""Return true if compile with Zoom support."""
    return hasattr(torch._C, "_zoom_getDeviceCount")

def is_available() -> bool:
    r"""Return a bool indicating if Zoom is currently available."""
    if not _is_compiled():
        return False
    return torch._C._zoom_getDeviceCount() > 0

def is_bf16_supported():
    r"""bfloat16 is supported on AMD GPU Archs"""
    return True

def is_initialized():
    r"""Return whether PyTorch's HIP state has been initialized."""
    return _initialized and not _is_in_bad_fork()

def init():
    r"""Initialize PyTorch's HIP state.

    You may need to call this explicitly if you are interacting with
    PyTorch via its C API, as Python bindings for Zoom functionality
    will not be available until this initialization takes place.

    No-op if Zoom is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize Zoom in forked subprocess. To use Zoom with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not hasattr(torch._C, "_zoom_getDeviceCount"):
            raise AssertionError("Torch not compiled with Zoom enabled")
        # if _cudart is None:
        #     raise AssertionError(
        #         "libcudart functions unavailable. It looks like you have a broken build?"
        #     )
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs
        # if "CUDA_MODULE_LOADING" not in os.environ:
        #     os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        torch._C._zoom_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        for calls in _lazy_seed_tracker.get_calls():
            if calls:
                _queued_calls.append(calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"Zoom call failed lazily at initialization with error: {str(e)}\n\n"
                        f"Zoom call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredZoomCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True




def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return torch._C._zoom_getDevice()

def device_count() -> int:
    r"""Return the number of GPUs available."""
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    r = torch._C._zoom_getDeviceCount()
    # NB: Do not cache the device count prior to Zoom initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to Zoom initialization.
    if _initialized:
        _cached_device_count = r
    return r


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    _lazy_init()
    return torch._C._zoom_getCurrentBlasHandle()


def set_sync_debug_mode(debug_mode: Union[int, str]) -> None:
    r"""Set the debug mode for zoom synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error. In
        particular, operations in torch.distributed and torch.sparse namespaces are not covered yet.
    """
    _lazy_init()
    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )

    torch._C._zoom_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode() -> int:
    r"""Return current value of debug mode for zoom synchronizing operations."""
    _lazy_init()
    return torch._C._zoom_get_sync_debug_mode()


################################################################################
# Define Storage and Tensor classes
################################################################################


@staticmethod  # type: ignore[misc]
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _ZoomBase.__new__
    return super(_ZoomBase, cls).__new__(cls, *args, **kwargs)


class _ZoomBase:
    is_zoom = True
    is_sparse = False

    def type(self, *args, **kwargs):
        # We could use a Protocol here to tell mypy that self has `get_device` method
        # but it is only available in the typing module on Python >= 3.8
        # or on typing_extensions module on Python >= 3.6
        with device(self.get_device()):  # type: ignore[attr-defined]
            return super().type(*args, **kwargs)  # type: ignore[misc]

    __new__ = _lazy_new


from torch.storage import _LegacyStorage, _warn_typed_storage_removal


class _ZoomLegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError("from_buffer: Not available for Zoom storage")

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError("_new_with_weak_ptr: Not available for Zoom storage")

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError("_new_shared_filename: Not available for Zoom storage")


class ByteStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


class ComplexDoubleStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cdouble


class ComplexFloatStorage(_ZoomLegacyStorage):
    @classproperty
    def dtype(self):
        _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.cfloat


del _LegacyStorage
del _ZoomLegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
torch._storage_classes.add(ComplexDoubleStorage)
torch._storage_classes.add(ComplexFloatStorage)
