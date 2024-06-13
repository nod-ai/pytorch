#pragma once
#include "../extension.h"

namespace at::detail {

    Tensor empty_generic_zoom(IntArrayRef size, ScalarType dtype, std::optional<Device> device, std::optional<c10::MemoryFormat> memory_format);
    Tensor empty_memory_format_zoom(IntArrayRef size, ::std::optional<ScalarType> dtype, ::std::optional<Layout> layout, ::std::optional<Device> device, ::std::optional<bool> pin_memory, ::std::optional<MemoryFormat> memory_format); // {"schema": "aten::empty.memory_format(SymInt[] size, *, ScalarTy
    
    Tensor empty_strided_generic_zoom(IntArrayRef size, IntArrayRef stride, ScalarType dtype, ::std::optional<Device> device_opt); 
    Tensor empty_strided_zoom(IntArrayRef size, IntArrayRef stride, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt); // {"schema": "aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "dispatch": "True", "default": "False"}

}