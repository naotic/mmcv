// Copyright (c) OpenMMLab. All rights reserved.
#include <vector>

#include "pytorch_cuda_helper.hpp"
#include "sca_attention_scatter_musa_kernel.muh"

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

void SCAAttentionScatterForwardMUSAKernelLauncher(
    const Tensor &queries, const std::vector<Tensor> &indexes, Tensor &slots) {
  c10::musa::MUSAGuard device_guard(queries.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = queries.size(0);
  const int num_cams = queries.size(1);
  const int max_len = queries.size(2);
  const int channels = queries.size(3);
  const int num_query = slots.size(1);
  dim3 threads(THREADS_PER_BLOCK);

  if (queries.scalar_type() == at::ScalarType::Float && channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *queries_data =
        reinterpret_cast<const float4 *>(queries.data_ptr<float>());
    float4 *slots_data = reinterpret_cast<float4 *>(slots.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_attention_scatter_forward_float4_musa_kernel<<<blocks, threads, 0,
                                                         stream>>>(
          bs, num_cams, max_len, num_query, channels4, valid_len, cam_id,
          queries_data, index.data_ptr<int64_t>(), slots_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      queries.scalar_type(), "sca_attention_scatter_forward_musa_kernel", [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_attention_scatter_forward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, max_len, num_query, channels, valid_len, cam_id,
                  queries.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                  slots.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SCAAttentionScatterDivForwardMUSAKernelLauncher(
    const Tensor &queries, const std::vector<Tensor> &indexes,
    const Tensor &count, Tensor &slots) {
  c10::musa::MUSAGuard device_guard(queries.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = queries.size(0);
  const int num_cams = queries.size(1);
  const int max_len = queries.size(2);
  const int channels = queries.size(3);
  const int num_query = slots.size(1);
  dim3 threads(THREADS_PER_BLOCK);

  if (queries.scalar_type() == at::ScalarType::Float && channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *queries_data =
        reinterpret_cast<const float4 *>(queries.data_ptr<float>());
    float4 *slots_data = reinterpret_cast<float4 *>(slots.data_ptr<float>());
    const float *count_data = count.data_ptr<float>();
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_attention_scatter_div_forward_float4_musa_kernel<<<blocks, threads,
                                                             0, stream>>>(
          bs, num_cams, max_len, num_query, channels4, valid_len, cam_id,
          queries_data, index.data_ptr<int64_t>(), count_data, slots_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      queries.scalar_type(), "sca_attention_scatter_div_forward_musa_kernel",
      [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_attention_scatter_div_forward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, max_len, num_query, channels, valid_len, cam_id,
                  queries.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                  count.data_ptr<scalar_t>(), slots.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SCAAttentionScatterBackwardMUSAKernelLauncher(
    const Tensor &grad_slots, const std::vector<Tensor> &indexes,
    Tensor &grad_queries) {
  c10::musa::MUSAGuard device_guard(grad_slots.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = grad_queries.size(0);
  const int num_cams = grad_queries.size(1);
  const int max_len = grad_queries.size(2);
  const int channels = grad_queries.size(3);
  const int num_query = grad_slots.size(1);
  dim3 threads(THREADS_PER_BLOCK);

  if (grad_slots.scalar_type() == at::ScalarType::Float && channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *grad_slots_data =
        reinterpret_cast<const float4 *>(grad_slots.data_ptr<float>());
    float4 *grad_queries_data =
        reinterpret_cast<float4 *>(grad_queries.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_attention_scatter_backward_float4_musa_kernel<<<blocks, threads, 0,
                                                          stream>>>(
          bs, num_cams, max_len, num_query, channels4, valid_len, cam_id,
          grad_slots_data, index.data_ptr<int64_t>(), grad_queries_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_slots.scalar_type(), "sca_attention_scatter_backward_musa_kernel",
      [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_attention_scatter_backward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, max_len, num_query, channels, valid_len, cam_id,
                  grad_slots.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                  grad_queries.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SCAAttentionScatterDivBackwardMUSAKernelLauncher(
    const Tensor &grad_slots, const std::vector<Tensor> &indexes,
    const Tensor &count, Tensor &grad_queries) {
  c10::musa::MUSAGuard device_guard(grad_slots.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = grad_queries.size(0);
  const int num_cams = grad_queries.size(1);
  const int max_len = grad_queries.size(2);
  const int channels = grad_queries.size(3);
  const int num_query = grad_slots.size(1);
  dim3 threads(THREADS_PER_BLOCK);

  if (grad_slots.scalar_type() == at::ScalarType::Float && channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *grad_slots_data =
        reinterpret_cast<const float4 *>(grad_slots.data_ptr<float>());
    float4 *grad_queries_data =
        reinterpret_cast<float4 *>(grad_queries.data_ptr<float>());
    const float *count_data = count.data_ptr<float>();
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_attention_scatter_div_backward_float4_musa_kernel<<<blocks, threads,
                                                              0, stream>>>(
          bs, num_cams, max_len, num_query, channels4, valid_len, cam_id,
          grad_slots_data, index.data_ptr<int64_t>(), count_data,
          grad_queries_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_slots.scalar_type(),
      "sca_attention_scatter_div_backward_musa_kernel", [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_attention_scatter_div_backward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, max_len, num_query, channels, valid_len, cam_id,
                  grad_slots.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                  count.data_ptr<scalar_t>(), grad_queries.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

Tensor sca_attention_scatter_forward(const Tensor &queries,
                                     const std::vector<Tensor> &indexes,
                                     const int num_query) {
  Tensor slots = queries.new_zeros({queries.size(0), num_query, queries.size(3)});
  SCAAttentionScatterForwardMUSAKernelLauncher(queries, indexes, slots);
  return slots;
}

Tensor sca_attention_scatter_div_forward(const Tensor &queries,
                                         const std::vector<Tensor> &indexes,
                                         const Tensor &count,
                                         const int num_query) {
  Tensor slots = queries.new_zeros({queries.size(0), num_query, queries.size(3)});
  Tensor count_contiguous = count.contiguous();
  SCAAttentionScatterDivForwardMUSAKernelLauncher(queries, indexes,
                                                  count_contiguous, slots);
  return slots;
}

Tensor sca_attention_scatter_backward(const Tensor &grad_slots,
                                      const std::vector<Tensor> &indexes,
                                      const int num_cams, const int max_len) {
  Tensor grad_queries =
      grad_slots.new_zeros({grad_slots.size(0), num_cams, max_len,
                            grad_slots.size(2)});
  SCAAttentionScatterBackwardMUSAKernelLauncher(grad_slots, indexes,
                                               grad_queries);
  return grad_queries;
}

Tensor sca_attention_scatter_div_backward(const Tensor &grad_slots,
                                          const std::vector<Tensor> &indexes,
                                          const Tensor &count,
                                          const int num_cams,
                                          const int max_len) {
  Tensor grad_queries =
      grad_slots.new_zeros({grad_slots.size(0), num_cams, max_len,
                            grad_slots.size(2)});
  Tensor count_contiguous = count.contiguous();
  SCAAttentionScatterDivBackwardMUSAKernelLauncher(
      grad_slots, indexes, count_contiguous, grad_queries);
  return grad_queries;
}

void SCAQueryGatherForwardMUSAKernelLauncher(
    const Tensor &query, const std::vector<Tensor> &indexes,
    Tensor &queries_rebatch) {
  c10::musa::MUSAGuard device_guard(query.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = query.size(0);
  const int num_query = query.size(1);
  const int channels = query.size(2);
  const int num_cams = queries_rebatch.size(1);
  const int max_len = queries_rebatch.size(2);
  dim3 threads(THREADS_PER_BLOCK);

  if (query.scalar_type() == at::ScalarType::Float && channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *query_data =
        reinterpret_cast<const float4 *>(query.data_ptr<float>());
    float4 *queries_rebatch_data =
        reinterpret_cast<float4 *>(queries_rebatch.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_query_gather_forward_float4_musa_kernel<<<blocks, threads, 0,
                                                    stream>>>(
          bs, num_cams, num_query, max_len, channels4, valid_len, cam_id,
          query_data, index.data_ptr<int64_t>(), queries_rebatch_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      query.scalar_type(), "sca_query_gather_forward_musa_kernel", [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_query_gather_forward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, num_query, max_len, channels, valid_len, cam_id,
                  query.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
                  queries_rebatch.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SCAQueryGatherBackwardMUSAKernelLauncher(
    const Tensor &grad_queries_rebatch, const std::vector<Tensor> &indexes,
    Tensor &grad_query) {
  c10::musa::MUSAGuard device_guard(grad_queries_rebatch.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = grad_query.size(0);
  const int num_query = grad_query.size(1);
  const int channels = grad_query.size(2);
  const int num_cams = grad_queries_rebatch.size(1);
  const int max_len = grad_queries_rebatch.size(2);
  dim3 threads(THREADS_PER_BLOCK);

  if (grad_queries_rebatch.scalar_type() == at::ScalarType::Float &&
      channels % 4 == 0) {
    const int channels4 = channels / 4;
    const float4 *grad_queries_rebatch_data =
        reinterpret_cast<const float4 *>(grad_queries_rebatch.data_ptr<float>());
    float4 *grad_query_data =
        reinterpret_cast<float4 *>(grad_query.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_query_gather_backward_float4_musa_kernel<<<blocks, threads, 0,
                                                     stream>>>(
          bs, num_cams, num_query, max_len, channels4, valid_len, cam_id,
          grad_queries_rebatch_data, index.data_ptr<int64_t>(),
          grad_query_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_queries_rebatch.scalar_type(),
      "sca_query_gather_backward_musa_kernel", [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * channels, THREADS_PER_BLOCK));
          sca_query_gather_backward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, num_query, max_len, channels, valid_len, cam_id,
                  grad_queries_rebatch.data_ptr<scalar_t>(),
                  index.data_ptr<int64_t>(), grad_query.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

Tensor sca_query_gather_forward(const Tensor &query,
                                const std::vector<Tensor> &indexes,
                                const int num_cams, const int max_len) {
  Tensor queries_rebatch =
      query.new_zeros({query.size(0), num_cams, max_len, query.size(2)});
  SCAQueryGatherForwardMUSAKernelLauncher(query, indexes, queries_rebatch);
  return queries_rebatch;
}

Tensor sca_query_gather_backward(const Tensor &grad_queries_rebatch,
                                 const std::vector<Tensor> &indexes,
                                 const int num_query) {
  Tensor grad_query = grad_queries_rebatch.new_zeros(
      {grad_queries_rebatch.size(0), num_query, grad_queries_rebatch.size(3)});
  SCAQueryGatherBackwardMUSAKernelLauncher(grad_queries_rebatch, indexes,
                                          grad_query);
  return grad_query;
}

void SCAQueryReferenceGatherForwardMUSAKernelLauncher(
    const Tensor &query, const Tensor &reference_points_cam,
    const std::vector<Tensor> &indexes, Tensor &queries_rebatch,
    Tensor &reference_points_rebatch) {
  c10::musa::MUSAGuard device_guard(query.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = query.size(0);
  const int num_query = query.size(1);
  const int channels = query.size(2);
  const int num_cams = queries_rebatch.size(1);
  const int max_len = queries_rebatch.size(2);
  const int depth2 = reference_points_rebatch.size(3) *
                     reference_points_rebatch.size(4);
  dim3 threads(THREADS_PER_BLOCK);

  if (query.scalar_type() == at::ScalarType::Float && channels % 4 == 0 &&
      depth2 % 4 == 0) {
    const int channels4 = channels / 4;
    const int depth2_4 = depth2 / 4;
    const float4 *query_data =
        reinterpret_cast<const float4 *>(query.data_ptr<float>());
    const float4 *reference_points_cam_data =
        reinterpret_cast<const float4 *>(reference_points_cam.data_ptr<float>());
    float4 *queries_rebatch_data =
        reinterpret_cast<float4 *>(queries_rebatch.data_ptr<float>());
    float4 *reference_points_rebatch_data =
        reinterpret_cast<float4 *>(reference_points_rebatch.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * channels4, THREADS_PER_BLOCK));
      sca_query_reference_gather_forward_float4_musa_kernel<<<blocks, threads,
                                                              0, stream>>>(
          bs, num_cams, num_query, max_len, channels4, depth2_4, valid_len,
          cam_id, query_data, reference_points_cam_data,
          index.data_ptr<int64_t>(), queries_rebatch_data,
          reference_points_rebatch_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      query.scalar_type(), "sca_query_reference_gather_forward_musa_kernel",
      [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          const int stride = channels > depth2 ? channels : depth2;
          dim3 blocks(GET_BLOCKS(bs * valid_len * stride, THREADS_PER_BLOCK));
          sca_query_reference_gather_forward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, num_query, max_len, channels, depth2,
                  valid_len, cam_id, query.data_ptr<scalar_t>(),
                  reference_points_cam.data_ptr<scalar_t>(),
                  index.data_ptr<int64_t>(),
                  queries_rebatch.data_ptr<scalar_t>(),
                  reference_points_rebatch.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void SCAReferenceGatherBackwardMUSAKernelLauncher(
    const Tensor &grad_reference_points_rebatch,
    const std::vector<Tensor> &indexes, Tensor &grad_reference_points_cam) {
  c10::musa::MUSAGuard device_guard(grad_reference_points_rebatch.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  const int bs = grad_reference_points_rebatch.size(0);
  const int num_cams = grad_reference_points_rebatch.size(1);
  const int max_len = grad_reference_points_rebatch.size(2);
  const int depth2 = grad_reference_points_rebatch.size(3) *
                     grad_reference_points_rebatch.size(4);
  const int num_query = grad_reference_points_cam.size(2);
  dim3 threads(THREADS_PER_BLOCK);

  if (grad_reference_points_rebatch.scalar_type() == at::ScalarType::Float &&
      depth2 % 4 == 0) {
    const int depth2_4 = depth2 / 4;
    const float4 *grad_reference_points_rebatch_data =
        reinterpret_cast<const float4 *>(
            grad_reference_points_rebatch.data_ptr<float>());
    float4 *grad_reference_points_cam_data =
        reinterpret_cast<float4 *>(grad_reference_points_cam.data_ptr<float>());
    for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
      const Tensor &index = indexes[cam_id];
      const int valid_len = index.numel();
      if (valid_len == 0) continue;
      dim3 blocks(GET_BLOCKS(bs * valid_len * depth2_4, THREADS_PER_BLOCK));
      sca_reference_gather_backward_float4_musa_kernel<<<blocks, threads, 0,
                                                         stream>>>(
          bs, num_cams, num_query, max_len, depth2_4, valid_len, cam_id,
          grad_reference_points_rebatch_data, index.data_ptr<int64_t>(),
          grad_reference_points_cam_data);
    }
    AT_MUSA_CHECK(musaGetLastError());
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_reference_points_rebatch.scalar_type(),
      "sca_reference_gather_backward_musa_kernel", [&] {
        for (int cam_id = 0; cam_id < num_cams; ++cam_id) {
          const Tensor &index = indexes[cam_id];
          const int valid_len = index.numel();
          if (valid_len == 0) continue;
          dim3 blocks(GET_BLOCKS(bs * valid_len * depth2, THREADS_PER_BLOCK));
          sca_reference_gather_backward_musa_kernel<scalar_t>
              <<<blocks, threads, 0, stream>>>(
                  bs, num_cams, num_query, max_len, depth2, valid_len, cam_id,
                  grad_reference_points_rebatch.data_ptr<scalar_t>(),
                  index.data_ptr<int64_t>(),
                  grad_reference_points_cam.data_ptr<scalar_t>());
        }
      });

  AT_MUSA_CHECK(musaGetLastError());
}

std::vector<Tensor> sca_query_reference_gather_forward(
    const Tensor &query, const Tensor &reference_points_cam,
    const std::vector<Tensor> &indexes, const int num_cams,
    const int max_len) {
  Tensor reference_points_cam_contiguous = reference_points_cam.contiguous();
  Tensor queries_rebatch =
      query.new_zeros({query.size(0), num_cams, max_len, query.size(2)});
  Tensor reference_points_rebatch = reference_points_cam.new_zeros(
      {query.size(0), num_cams, max_len, reference_points_cam.size(3),
       reference_points_cam.size(4)});
  SCAQueryReferenceGatherForwardMUSAKernelLauncher(
      query, reference_points_cam_contiguous, indexes, queries_rebatch,
      reference_points_rebatch);
  return {queries_rebatch, reference_points_rebatch};
}

Tensor sca_reference_gather_backward(
    const Tensor &grad_reference_points_rebatch,
    const std::vector<Tensor> &indexes, const int num_query) {
  Tensor grad_reference_points_cam = grad_reference_points_rebatch.new_zeros(
      {grad_reference_points_rebatch.size(1),
       grad_reference_points_rebatch.size(0), num_query,
       grad_reference_points_rebatch.size(3),
       grad_reference_points_rebatch.size(4)});
  SCAReferenceGatherBackwardMUSAKernelLauncher(
      grad_reference_points_rebatch.contiguous(), indexes,
      grad_reference_points_cam);
  return grad_reference_points_cam;
}
