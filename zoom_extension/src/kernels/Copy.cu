// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include "../CachingHostAllocator.h"
#include "../ZoomContext.h"
#include "../ZoomEvent.h"
#include "../PeerToPeerAccess.h"
#include <ATen/native/Copy.h>
#include <ATen/native/quantized/Copy.h>
#include <ATen/native/TensorIterator.h>
#include "../jit/Loops.cuh"
#include <ATen/quantized/Quantizer.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include "../ZoomCachingAllocator.h"
#include "../ZoomStream.h"
#include "Copy.h"

// TODO(NS): Investigate why FP8 conversion intrinsics end up being slower
// #ifdef AT_USE_NV_CVT_INTRINSICS
// #include <cuda_fp8.h>
// #endif

namespace at::native {

  // copied these copy functions from native/Copy.cpp for simplicity, can remove once we use DispatchStub
  // rather than overriding at::_copy_from
  #define _AT_DISPATCH_CP_TYPES(TYPE, NAME, ...)                              \
        AT_DISPATCH_V2(                             \
            TYPE, NAME, AT_WRAP(__VA_ARGS__), kComplexHalf, kHalf, kBool, kBFloat16, kFloat8_e5m2,            \
            kFloat8_e4m3fn, kFloat8_e5m2fnuz, kFloat8_e4m3fnuz, AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES))
  
  bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  const int MIN_SZ = 60 * 60;
  return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
      src.stride(0) == 1 && src.stride(1) == src.size(0) &&
      self.scalar_type() == src.scalar_type() &&
      !isBitsType(self.scalar_type()) &&
      self.sizes().equals(src.sizes()) &&
      self.is_neg() == src.is_neg() &&
      self.is_conj() == src.is_conj() &&
      self.numel() >= MIN_SZ;
  }

  void copy_same_type_transpose_(Tensor& self, const Tensor& src) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t BLOCK_SZ;
  if (self.scalar_type() == kByte) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 120;
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    BLOCK_SZ = 60;
  }
  Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

  // The code below is implemented with the assumption that sizes are equal
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.sizes().equals(src.sizes()));

  _AT_DISPATCH_CP_TYPES(self.scalar_type(), "copy_", [&] {
    const scalar_t* sp = src.const_data_ptr<scalar_t>();
    scalar_t* rp = self.data_ptr<scalar_t>();
    scalar_t* bp = buf.data_ptr<scalar_t>();

    int64_t NR = src.size(0);
    int64_t NC = src.size(1);
    for (int64_t R = 0; R < NR; R += BLOCK_SZ) {
      for (int64_t C = 0; C < NC; C += BLOCK_SZ) {
        const scalar_t* spo = sp + R + C * NR;
        scalar_t* rpo = rp + C + R * NC;

        int nr = std::min(NR - R, BLOCK_SZ);
        int nc = std::min(NC - C, BLOCK_SZ);

        // 1. copy columns from src to buf
        for (const auto c : c10::irange(nc)) {
          memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(scalar_t));
        }

        // 2. transpose buf in place
        int rc_max = std::max(nr, nc);
        int rc_min = std::min(nr, nc);
        for (const auto r : c10::irange(rc_max)) {
          int end = std::min(r, rc_min);
          for (const auto c : c10::irange(end)) {
            scalar_t tmp = bp[r + BLOCK_SZ * c];
            bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
            bp[r * BLOCK_SZ + c] = tmp;
          }
        }

        // 3. copy rows from buf to dst
        for (const auto r : c10::irange(nr)) {
          memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(scalar_t));
        }
      }
    }
  });
}

void neg_kernel_zoom(TensorIteratorBase &iter);
void conj_kernel_zoom(TensorIteratorBase &iter);

void float8_copy_kernel_zoom(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  ScalarType other_dtype = iter.dtype(1);
  if (dtype == kFloat8_e4m3fn) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fn(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fn(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fn x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_float_to_fp8(value, __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_halfraw_to_fp8(static_cast<__half>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
#ifdef AT_USE_NV_CVT_INTRINSICS
             const auto x =  __nv_cvt_bfloat16raw_to_fp8(static_cast<__nv_bfloat16>(value), __NV_NOSAT, __NV_E5M2);
             return Float8_e5m2(x, Float8_e5m2::from_bits());
#else
             return Float8_e5m2(value);
#endif
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2 x) { return x; });
         break;
    }
  } else if (dtype == kFloat8_e4m3fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e4m3fnuz(value);
         });
         break;
      default:
        gpu_kernel(iter, [] GPU_LAMBDA(Float8_e4m3fnuz x) { return x; });
        break;
    }
  } else if (dtype == kFloat8_e5m2fnuz) {
    switch (other_dtype) {
      case kFloat:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(float value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kHalf:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(Half value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      case kBFloat16:
         gpu_kernel_nocast(iter, [] GPU_LAMBDA(BFloat16 value) {
             return Float8_e5m2fnuz(value);
         });
         break;
      default:
         gpu_kernel(iter, [] GPU_LAMBDA(Float8_e5m2fnuz x) { return x; });
         break;
    }
  } else {
    TORCH_CHECK(false, "This supposed ot be called only for Float8 types");
  }
}

// TODO: We probably can use the opaque type trick to avoid creating duplicate
// kernels for equivalent bit lengths
void direct_copy_kernel_zoom(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    AT_DISPATCH_QINT_TYPES(dtype, "copy_", [&] {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else if (dtype == kFloat8_e5m2 || dtype == kFloat8_e4m3fn || dtype == kFloat8_e5m2fnuz || dtype == kFloat8_e4m3fnuz) {
     float8_copy_kernel_zoom(iter);
  } else if (isBitsType(dtype)) {
    TORCH_CHECK(dtype == iter.dtype(1), "copy_() does not support casting "
      "bits types to different bits types. Source dtype is ", iter.dtype(1), "target dtype is ", dtype);
    AT_DISPATCH_BIT_TYPES(dtype, "copy_", [&] {
      gpu_kernel_nocast(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    });
  } else {
    AT_DISPATCH_V2(
        dtype, "copy_", AT_WRAP([&] {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return x; });
    }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kHalf, kBool, kBFloat16, kComplexHalf, AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void neg_conj_kernel_zoom(TensorIteratorBase &iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_zoom", [&] {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t x) { return -std::conj(x); });
  });
}

using namespace at::zoom;

// device-to-device copy, does type conversion
void copy_device_to_device(TensorIterator& iter,
                           bool non_blocking,
                           bool p2p_enabled) {
  int64_t numel = iter.numel();

  // We can memcpy the memory if both tensors have the same type AND both
  // tensors are contiguous after dimension coalescing and reordering.
  bool same_type = iter.dtype(0) == iter.dtype(1);
  bool same_conj = iter.tensor(0).is_conj() == iter.tensor(1).is_conj();
  bool same_neg = iter.tensor(0).is_neg() == iter.tensor(1).is_neg();
  bool memcpy_eligible = same_type && same_conj && same_neg && iter.is_contiguous();

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  c10::zoom::ZoomGuard device_guard(src_device);

  // We always perform the copy on the source device, using the current stream
  // on the source device, and we fully synchronize on both src and dst's
  // current streams for completion of the copy. We have to explicitly do this
  // for non-contig copies. This mimics the behavior of cross-device
  // hipMemcpyAsync on the default stream.
  c10::zoom::ZoomStream copy_stream = c10::zoom::getCurrentZoomStream(src_device.index());
  if (src_device != dst_device) {
    // This is a cross-device copy on the src current stream and dst current
    // stream. We perform a two-way barrier between both devices' streams
    // before the copy. This ensures that any write-after-write and
    // write-after-read dependencies on the destination side are handled, so
    // that no one is operating on the dst memory when we perform the copy.
    // src waits on dst barrier (src already waits on src)
    ZoomEvent dst_ready;
    device_guard.set_device(dst_device);
    dst_ready.record(c10::zoom::getCurrentZoomStream(dst_device.index()));

    device_guard.set_device(src_device);
    dst_ready.block(copy_stream);
  }

  if (memcpy_eligible) {
    void *dst = iter.data_ptr(0);
    void *src = iter.data_ptr(1);
    size_t size = numel * iter.element_size(0);
    if (src != dst || src_device != dst_device) {
      // Due to bizarre cuda driver intricacies, copies of
      // hipMallocAsynced memory between devices that aren't
      // peer-to-peer-capable need "hipMemcpyPeerAsync".
      // So we let the allocator implement the correct call
      // (either hipMemcpyAsync or hipMemcpyPeerAsync)
      C10_ZOOM_CHECK(c10::zoom::ZoomCachingAllocator::memcpyAsync(
        dst, dst_device.index(),
        src, src_device.index(),
        size, copy_stream, p2p_enabled));
    }
  } else {
    if (same_neg) {
      if (!same_conj) {
        conj_kernel_zoom(iter);
      } else {
        direct_copy_kernel_zoom(iter);
      }
    } else {
      if (!same_conj) {
        neg_conj_kernel_zoom(iter);
      } else {
        neg_kernel_zoom(iter);
      }
    }
  }

  if (src_device != dst_device) {
    // dst waits on src barrier (dst already waits on dst). We cannot
    // operate on dst's copy until the copy is complete.

    // Still on src_device, record stream event
    ZoomEvent src_ready;
    src_ready.record(copy_stream);

    device_guard.set_device(dst_device);
    src_ready.block(c10::zoom::getCurrentZoomStream(dst_device.index()));
  }

  C10_ZOOM_CHECK(hipGetLastError());
}

static bool copy_requires_temporaries(TensorIterator& iter, bool p2p_enabled) {
  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  if (dst_device == src_device) {
    // We never require temporaries for copies on the same GPU.
    TORCH_INTERNAL_ASSERT(dst_device.is_privateuseone() && src_device.is_privateuseone());
    return false;
  }

  bool same_dtype = iter.dtype(0) == iter.dtype(1);
  if (same_dtype && iter.is_contiguous()) {
    // Contiguous same-dtype copies can always use hipMemcpyAsync
    return false;
  } else if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    // Copies between GPUs can use the copy kernel if P2P is supported
    return !p2p_enabled;
  } else {
    // The remaining cases require temporaries. For example, this includes
    // non-contiguous copies between CPU and GPU.
    return true;
  }
}

static bool maybe_enable_p2p_access(Device dst_device, Device src_device) {
  if (dst_device.is_cpu() || src_device.is_cpu()) {
    return false;
  }
  return at::zoom::get_p2p_access(src_device.index(), dst_device.index());
}

static void copy_kernel_zoom(TensorIterator& iter, bool non_blocking) {
  TORCH_CHECK(iter.ntensors() == 2);

  Device dst_device = iter.device(0);
  Device src_device = iter.device(1);

  // Enable p2p access between devices. (No-op if it involves the CPU)
  bool p2p_enabled = maybe_enable_p2p_access(dst_device, src_device);

  if (copy_requires_temporaries(iter, p2p_enabled)) {
    // NB: this involves recursive calls to copy. Be careful that those copies
    // don't require temporaries or you will cause an infinite recursion!
    auto& dst = iter.tensor(0);
    Tensor dst_contig;
    Tensor src_contig;

    // If non_blocking is true - type conversions are performed on the GPU
    // For blocking transfers conversions are performed on CPU to avoid allocating
    // extra GPU memory
    // for GPU-GPU transfers conversions are performed on the source device
    auto conversion_device = non_blocking ? DeviceType::PrivateUse1 : kCPU;
    if (iter.device_type(1) == conversion_device) {
      dst_contig = dst.is_contiguous() ? dst : at::empty_like(dst, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).to(iter.dtype(0)).expand_as(dst).contiguous();
    } else {
      bool same_type = iter.dtype(0) == iter.dtype(1);
      dst_contig = (dst.is_contiguous() && same_type) ? dst : at::empty_like(dst, iter.dtype(1), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      src_contig = iter.tensor(1).expand_as(dst).contiguous();
    }

    // propagate the correct conjugate bit
    dst_contig._set_conj(dst.is_conj());
    src_contig._set_conj(iter.tensor(1).is_conj());

    dst_contig._set_neg(dst.is_neg());
    src_contig._set_neg(iter.tensor(1).is_neg());

    // perform a same-dtype copy on contiguous tensors
    TORCH_INTERNAL_ASSERT(dst_contig.sizes().equals(src_contig.sizes()));
    TORCH_INTERNAL_ASSERT(dst_contig.scalar_type() == src_contig.scalar_type());
    dst_contig.copy_(src_contig, non_blocking);

    // if necessary, copy back into dst
    if (!dst_contig.is_same(dst)) {
      TORCH_INTERNAL_ASSERT(dst_contig.device() == dst.device());
      dst.copy_(dst_contig, non_blocking);
    }
    return;
  }

  // Copy on GPU (or between GPUs)
  if (dst_device.is_privateuseone() && src_device.is_privateuseone()) {
    copy_device_to_device(iter, non_blocking, p2p_enabled);
    return;
  }

  // Copy between CPU and GPU
  c10::zoom::OptionalZoomGuard device_guard;
  hipMemcpyKind kind;
  if (dst_device.is_privateuseone() && src_device.is_cpu()) {
    device_guard.set_device(dst_device);
    kind = hipMemcpyHostToDevice;
  } else if (dst_device.is_cpu() && src_device.is_privateuseone()) {
    device_guard.set_device(src_device);
    kind = hipMemcpyDeviceToHost;
  } else {
    TORCH_INTERNAL_ASSERT(false, "unsupported devices in GPU copy_()");
  }

  void* dst = iter.data_ptr(0);
  void* src = iter.data_ptr(1);
  int64_t nbytes = iter.numel() * iter.element_size(0);
  c10::zoom::ZoomStream stream = c10::zoom::getCurrentZoomStream();

  if (non_blocking) {
    C10_ZOOM_CHECK(hipMemcpyAsync(dst, src, nbytes, kind, stream));
    // we use both the storage context and the tensor data pointer as the key
    // for the caching host allocator. This allows us to better attribute the
    // events to the original tensor allocation correctly. The cases we seek to
    // handle are:

    // 1: a user can pass a pinned memory tensor with an alternative
    // context, for example if allocating memory directly from the pinned memory
    // allocator and constructing a tensor with torch::from_blob.

    // 2: a user can pass a tensor with a different base pointer to the original
    // allocation (via slicing).
    const auto& dst_tensor = iter.tensor(0);
    const auto& src_tensor = iter.tensor(1);
    const auto& host_tensor = (dst_device == kCPU ? dst_tensor : src_tensor);
    auto* ptr = (dst_device == kCPU ? dst : src);
    auto* ctx = host_tensor.storage().data_ptr().get_context();
    // TODO: warn on the return value.
    CachingHostAllocator_recordEvent(ptr, ctx, stream);

  } else {
    c10::zoom::memcpy_and_sync(dst, src, nbytes, kind, stream);
  }

  if (iter.tensor(0).is_conj() != iter.tensor(1).is_conj()) {
     iter.tensor(0).conj_physical_();
  }
  if (iter.tensor(0).is_neg() != iter.tensor(1).is_neg()) {
     iter.tensor(0).neg_();
  }
}

    // Copy kernels only support overriding the registered DispatchStub for a hardcoded set of in-tree devices,
    // in principle we can do that rather than using TORCH_LIBRARY_IMPL if we add zoom as a 'supported device'
    // this is a TODO for once we're in-tree
    // REGISTER_PRIVATEUSE1_DISPATCH(copy_stub, &copy_kernel_zoom);

    Tensor& zoom_copy_(Tensor & self, const Tensor & src, bool non_blocking) {
        // copied sanity checks/routing from Copy.cpp 
        TORCH_CHECK(!(self.is_quantized() || src.is_quantized()), "Quantized Copy unimplemented in Zoom Backend");
        // if (self.is_quantized() && !src.is_quantized()) {
        //     return quantized_copy_from_float_(self, src);
        // }

        // if (self.is_quantized() && src.is_quantized()) {
        //     TORCH_CHECK(self.qscheme() == src.qscheme(),
        //                 "Quantized Copy only works with same qscheme");
        //     TORCH_CHECK(self.scalar_type() == src.scalar_type());
        //     set_quantizer_(self, src.quantizer());
        // }

        // if (!self.is_quantized() && src.is_quantized()) {
        //     TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
        // }

        const bool is_same_data = (
          self.is_alias_of(src) &&
          self.storage_offset() == src.storage_offset() &&
          self.strides().equals(src.strides()) &&
          self.sizes().equals(src.sizes()) &&
          self.scalar_type() == src.scalar_type() &&
          self.is_conj() == src.is_conj() &&
          self.is_neg() == src.is_neg()
        );
        
        if (is_same_data) {
            return self;
        }

        // copy from self to dst
        auto iter = TensorIteratorConfig()
            .add_output(self)
            .add_const_input(src)
            .resize_outputs(false)
            .check_all_same_dtype(false)
            .check_all_same_device(false)
            .build();

        if (iter.numel() == 0) {
            return self;
        }

        DeviceType device_type = iter.device_type(0);

        if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
            copy_same_type_transpose_(self, src);
            return self;
        }
        
        if(!(self.is_complex() || self.dtype() == at::kBool) && src.is_complex()) {
            TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
        }
        
        copy_kernel_zoom(iter, non_blocking);
        return self;

    }

    Tensor zoom_copy_from(const Tensor& self, const Tensor& dst, bool non_blocking) {
      return zoom_copy_(const_cast<Tensor&>(dst), self, non_blocking);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("copy_", &zoom_copy_);
        m.impl("_copy_from", &zoom_copy_from);
    }

} // namespace at::native