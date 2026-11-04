/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <musa.h>
#include <musa_runtime.h>

#include <THC/THCAtomics.muh>
#include <vector>

#include "ms_deform_attn_cuda_kernel.muh"

namespace {
constexpr int kMsDeformAttnNumThreads = 1024;
constexpr int kMsDeformAttnChannels32WarpsPerBlock = 8;
constexpr int kMsDeformAttnWarpSize = 32;

inline int GetMsDeformAttnBlocks(const int n, const int num_threads) {
  return (n + num_threads - 1) / num_threads;
}
}  // namespace

template <typename scalar_t>
void ms_deformable_im2col_musa(musaStream_t stream, const scalar_t *data_value,
                               const int64_t *data_spatial_shapes,
                               const int64_t *data_level_start_index,
                               const scalar_t *data_sampling_loc,
                               const scalar_t *data_attn_weight,
                               const int batch_size, const int spatial_size,
                               const int num_heads, const int channels,
                               const int num_levels, const int num_query,
                               const int num_point, scalar_t *data_col) {
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = kMsDeformAttnNumThreads;
  if (channels == 32 && num_levels == 1 && num_point == 4) {
    ms_deformable_im2col_gpu_kernel_channels32<scalar_t, 1, 4>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads,
           0, stream>>>(num_kernels, data_value, data_spatial_shapes,
                        data_level_start_index, data_sampling_loc,
                        data_attn_weight, batch_size, spatial_size, num_heads,
                        num_query, data_col);
  } else if (channels == 32 && num_levels == 4 && num_point == 8) {
    ms_deformable_im2col_gpu_kernel_channels32_runtime<scalar_t>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads,
           0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_heads, num_levels, num_query, num_point, data_col);
  } else {
    ms_deformable_im2col_gpu_kernel<scalar_t>
        <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads,
           0, stream>>>(
            num_kernels, data_value, data_spatial_shapes, data_level_start_index,
            data_sampling_loc, data_attn_weight, batch_size, spatial_size,
            num_heads, channels, num_levels, num_query, num_point, data_col);
  }

  musaError_t err = musaGetLastError();
  if (err != musaSuccess) {
    printf("error in ms_deformable_im2col_musa: %s\n", musaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_col2im_musa(
    musaStream_t stream, const scalar_t *grad_col, const scalar_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int batch_size, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int num_query,
    const int num_point, scalar_t *grad_value, scalar_t *grad_sampling_loc,
    scalar_t *grad_attn_weight) {
  const int num_threads =
      (channels > kMsDeformAttnNumThreads) ? kMsDeformAttnNumThreads : channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > kMsDeformAttnNumThreads) {
    if ((channels & (kMsDeformAttnNumThreads - 1)) == 0) {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads,
             num_threads * 3 * sizeof(scalar_t), stream>>>(
              num_kernels, grad_col, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight,
              batch_size, spatial_size, num_heads, channels, num_levels,
              num_query, num_point, grad_value, grad_sampling_loc,
              grad_attn_weight);
    } else {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
          <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads), num_threads,
             0,
             stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, spatial_size, num_heads,
                       channels, num_levels, num_query, num_point, grad_value,
                       grad_sampling_loc, grad_attn_weight);
    }
  } else {
    switch (channels) {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      1>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      2>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      4>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      8>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t,
                                                                      16>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_warp_broadcast_multi_warp_channels32<
            scalar_t, kMsDeformAttnChannels32WarpsPerBlock>
            <<<GetMsDeformAttnBlocks(num_actual_kernels / kMsDeformAttnWarpSize,
                                     kMsDeformAttnChannels32WarpsPerBlock),
               kMsDeformAttnChannels32WarpsPerBlock * kMsDeformAttnWarpSize, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      64>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      128>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      256>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      512>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t,
                                                                      1024>
            <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
               num_threads, 0,
               stream>>>(num_kernels, grad_col, data_value, data_spatial_shapes,
                         data_level_start_index, data_sampling_loc,
                         data_attn_weight, batch_size, spatial_size, num_heads,
                         channels, num_levels, num_query, num_point, grad_value,
                         grad_sampling_loc, grad_attn_weight);
        break;
      default:
        if (channels < 64) {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
              <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        } else {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
              <<<GetMsDeformAttnBlocks(num_actual_kernels, num_threads),
                 num_threads,
                 num_threads * 3 * sizeof(scalar_t), stream>>>(
                  num_kernels, grad_col, data_value, data_spatial_shapes,
                  data_level_start_index, data_sampling_loc, data_attn_weight,
                  batch_size, spatial_size, num_heads, channels, num_levels,
                  num_query, num_point, grad_value, grad_sampling_loc,
                  grad_attn_weight);
        }
    }
  }
  musaError_t err = musaGetLastError();
  if (err != musaSuccess) {
    printf("error in ms_deformable_col2im_musa: %s\n", musaGetErrorString(err));
  }
}

template <typename scalar_t>
void ms_deformable_im2col_scatter_div_musa(
    musaStream_t stream, const scalar_t *data_value,
    const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index,
    const scalar_t *data_sampling_loc, const scalar_t *data_attn_weight,
    const int64_t *data_index, const scalar_t *data_count,
    const int batch_size, const int num_cams, const int cam_id,
    const int valid_len, const int spatial_size, const int num_heads,
    const int channels, const int num_levels, const int max_len,
    const int num_point, const int num_query, scalar_t *data_slots) {
  if (valid_len == 0) {
    return;
  }
  const int num_kernels = batch_size * valid_len * num_heads * channels;
  const int num_threads = kMsDeformAttnNumThreads;
  if (channels == 32) {
    ms_deformable_im2col_scatter_div_gpu_kernel_channels32_runtime<scalar_t>
        <<<GetMsDeformAttnBlocks(num_kernels, num_threads), num_threads, 0,
           stream>>>(num_kernels, data_value, data_spatial_shapes,
                     data_level_start_index, data_sampling_loc,
                     data_attn_weight, data_index, data_count, batch_size,
                     num_cams, cam_id, spatial_size, num_heads, num_levels,
                     valid_len, max_len, num_point, num_query, data_slots);
  } else {
    printf("unsupported channels in ms_deformable_im2col_scatter_div_musa: %d\n",
           channels);
  }
}

at::Tensor ms_deform_attn_musa_forward(const at::Tensor &value,
                                       const at::Tensor &spatial_shapes,
                                       const at::Tensor &level_start_index,
                                       const at::Tensor &sampling_loc,
                                       const at::Tensor &attn_weight,
                                       const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");

  AT_ASSERTM(value.is_privateuseone(), "value must be a MUSA tensor");
  AT_ASSERTM(spatial_shapes.is_privateuseone(), "spatial_shapes must be a MUSA tensor");
  AT_ASSERTM(level_start_index.is_privateuseone(),
             "level_start_index must be a MUSA tensor");
  AT_ASSERTM(sampling_loc.is_privateuseone(), "sampling_loc must be a MUSA tensor");
  AT_ASSERTM(attn_weight.is_privateuseone(), "attn_weight must be a MUSA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  const int im2col_step_ = std::min(batch, im2col_step);

  AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)",
             batch, im2col_step_);

  auto output =
      at::empty({batch, num_query, num_heads, channels}, value.options());

  const int batch_n = im2col_step_;
  auto output_n = output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto columns = output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "ms_deform_attn_forward_musa", ([&] {
          ms_deformable_im2col_musa(
              c10::musa::getCurrentMUSAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              batch_n, spatial_size, num_heads, channels, num_levels, num_query,
              num_point, columns.data_ptr<scalar_t>());
        }));
  }

  output = output.view({batch, num_query, num_heads * channels});

  return output;
}

at::Tensor sca_ms_deform_attn_scatter_div_musa_forward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const std::vector<at::Tensor> &indexes,
    const at::Tensor &count, const int num_query) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(count.is_contiguous(), "count tensor has to be contiguous");

  AT_ASSERTM(value.is_privateuseone(), "value must be a MUSA tensor");
  AT_ASSERTM(spatial_shapes.is_privateuseone(),
             "spatial_shapes must be a MUSA tensor");
  AT_ASSERTM(level_start_index.is_privateuseone(),
             "level_start_index must be a MUSA tensor");
  AT_ASSERTM(sampling_loc.is_privateuseone(),
             "sampling_loc must be a MUSA tensor");
  AT_ASSERTM(attn_weight.is_privateuseone(),
             "attn_weight must be a MUSA tensor");
  AT_ASSERTM(count.is_privateuseone(), "count must be a MUSA tensor");

  const int batch_num_cams = value.size(0);
  const int num_cams = indexes.size();
  AT_ASSERTM(num_cams > 0, "indexes must not be empty");
  AT_ASSERTM(batch_num_cams % num_cams == 0,
             "value batch must be divisible by num_cams");
  const int batch = batch_num_cams / num_cams;
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);
  const int num_levels = spatial_shapes.size(0);
  const int max_len = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  auto slots = at::zeros({batch, num_query, num_heads * channels},
                         value.options());
  auto stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      value.scalar_type(), "sca_ms_deform_attn_scatter_div_musa_forward", ([&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const at::Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) {
            continue;
          }
          ms_deformable_im2col_scatter_div_musa(
              stream, value.data_ptr<scalar_t>(),
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>(), attn_weight.data_ptr<scalar_t>(),
              index.data_ptr<int64_t>(), count.data_ptr<scalar_t>(), batch,
              num_cams, cam_id, valid_len, spatial_size, num_heads, channels,
              num_levels, max_len, num_point, num_query,
              slots.data_ptr<scalar_t>());
        }
      }));

  musaError_t err = musaGetLastError();
  if (err != musaSuccess) {
    printf("error in sca_ms_deform_attn_scatter_div_musa_forward: %s\n",
           musaGetErrorString(err));
  }
  return slots;
}

void ms_deform_attn_musa_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, const int im2col_step) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc.is_contiguous(),
             "sampling_loc tensor has to be contiguous");
  AT_ASSERTM(attn_weight.is_contiguous(),
             "attn_weight tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  AT_ASSERTM(value.is_privateuseone(), "value must be a MUSA tensor");
  AT_ASSERTM(spatial_shapes.is_privateuseone(), "spatial_shapes must be a MUSA tensor");
  AT_ASSERTM(level_start_index.is_privateuseone(),
             "level_start_index must be a MUSA tensor");
  AT_ASSERTM(sampling_loc.is_privateuseone(), "sampling_loc must be a MUSA tensor");
  AT_ASSERTM(attn_weight.is_privateuseone(), "attn_weight must be a MUSA tensor");
  AT_ASSERTM(grad_output.is_privateuseone(), "grad_output must be a MUSA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);

  const int num_query = sampling_loc.size(1);
  const int num_point = sampling_loc.size(4);

  const int im2col_step_ = std::min(batch, im2col_step);

  AT_ASSERTM(batch % im2col_step_ == 0, "batch(%d) must divide im2col_step(%d)",
             batch, im2col_step_);

  const int batch_n = im2col_step_;
  auto per_value_size = spatial_size * num_heads * channels;
  auto per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2;
  auto per_attn_weight_size = num_query * num_heads * num_levels * num_point;
  auto grad_output_n = grad_output.view(
      {batch / im2col_step_, batch_n, num_query, num_heads, channels});

  for (int n = 0; n < batch / im2col_step_; ++n) {
    auto grad_output_g = grad_output_n.select(0, n);
    AT_DISPATCH_FLOATING_TYPES(
        value.scalar_type(), "ms_deform_attn_backward_musa", ([&] {
          ms_deformable_col2im_musa(
              c10::musa::getCurrentMUSAStream(),
              grad_output_g.data_ptr<scalar_t>(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data_ptr<int64_t>(),
              level_start_index.data_ptr<int64_t>(),
              sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size,
              batch_n, spatial_size, num_heads, channels, num_levels, num_query,
              num_point,
              grad_value.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_value_size,
              grad_sampling_loc.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_sample_loc_size,
              grad_attn_weight.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_attn_weight_size);
        }));
  }
}
