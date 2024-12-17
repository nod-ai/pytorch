#pragma once

#include <ATen/ATen.h>
#include <ATen/zoom/ATenZoomGeneral.h>
#include <ATen/zoom/ZoomContext.h>
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>

#include <cstddef>
#include <vector>

namespace torch::zoom {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

TORCH_ZOOM_API std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors);
TORCH_ZOOM_API std::vector<at::Tensor> broadcast(
    const at::Tensor& tensor,
    at::IntArrayRef devices);
TORCH_ZOOM_API tensor_list2d broadcast_coalesced(
    at::TensorList tensors,
    at::IntArrayRef devices,
    size_t buffer_size);

TORCH_ZOOM_API std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim = 0,
    const std::optional<std::vector<c10::optional<c10::zoom::ZoomStream>>>&
        streams = c10::nullopt);

TORCH_ZOOM_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const std::optional<std::vector<c10::optional<c10::zoom::ZoomStream>>>&
        streams = c10::nullopt);

TORCH_ZOOM_API at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim);

TORCH_ZOOM_API at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    std::optional<int32_t> destination_index);

} // namespace torch::zoom
