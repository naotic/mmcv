// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <stdio.h>

#include "iou3d_cuda_kernel.muh"
#include "pytorch_cuda_helper.hpp"

int const threadsPerBlock = sizeof(unsigned long long int) * 8;

inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

__global__ static void gather_keep_from_mask(bool *keep,
                                             const unsigned long long *dev_mask,
                                             const int n_boxes) {
  const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  const int tid = threadIdx.x;

  // mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // initialize removed.
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int nblock = 0; nblock < col_blocks; ++nblock) {
    auto removed_val = removed[nblock];
    __syncthreads();
    const int i_offset = nblock * threadsPerBlock;
#pragma unroll
    for (int inblock = 0; inblock < threadsPerBlock; ++inblock) {
      const int i = i_offset + inblock;
      if (i >= n_boxes) break;
      // select a candidate, check if it should kept.
      if (!(removed_val & (1ULL << inblock))) {
        if (tid == 0) {
          // mark the output.
          keep[i] = true;
        }
        auto p = dev_mask + i * col_blocks;
        // remove all bboxes which overlap the candidate.
        for (int j = tid; j < col_blocks; j += blockDim.x) {
          if (j >= nblock) removed[j] |= p[j];
        }
        __syncthreads();
        removed_val = removed[nblock];
      }
    }
  }
}

void IoU3DNMS3DForwardCUDAKernelLauncher(const Tensor boxes, Tensor& keep,
                                         Tensor& keep_num,
                                         float nms_overlap_thresh) {
  using namespace at::indexing;
  c10::musa::MUSAGuard device_guard(boxes.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  int boxes_num = boxes.size(0);

  const int col_blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(::at::musa::kMUSA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long*)mask.data_ptr<int64_t>(),
      boxes_num);

  auto keep_data = keep_t.nonzero().index({Slice(), 0});
  keep_num.fill_(at::Scalar(keep_data.size(0)));
  keep.index_put_({Slice(0, keep_data.size(0))}, keep_data);
  AT_MUSA_CHECK(musaGetLastError());
}

void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap) {
  at::musa::MUSAGuard device_guard(boxes_a.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK_IOU3D),
              DIVUP(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_overlap_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_overlap.data_ptr<float>());

  AT_MUSA_CHECK(musaGetLastError());
}

void IoU3DBoxesIoUBevForwardCUDAKernelLauncher(const int num_a,
                                               const Tensor boxes_a,
                                               const int num_b,
                                               const Tensor boxes_b,
                                               Tensor ans_iou) {
  at::musa::MUSAGuard device_guard(boxes_a.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK_IOU3D),
              DIVUP(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_iou_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_iou.data_ptr<float>());

  AT_MUSA_CHECK(musaGetLastError());
}

void IoU3DNMSForwardCUDAKernelLauncher(const Tensor boxes,
                                       unsigned long long *mask, int boxes_num,
                                       float nms_overlap_thresh) {
  at::musa::MUSAGuard device_guard(boxes.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_MUSA_CHECK(musaGetLastError());
}

void IoU3DNMSNormalForwardCUDAKernelLauncher(const Tensor boxes,
                                             unsigned long long *mask,
                                             int boxes_num,
                                             float nms_overlap_thresh) {
  at::musa::MUSAGuard device_guard(boxes.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_MUSA_CHECK(musaGetLastError());
}

void IoU3DNMS3DNormalForwardCUDAKernelLauncher(const Tensor boxes, Tensor& keep,
                                               Tensor& keep_num,
                                               float nms_overlap_thresh) {
  using namespace at::indexing;
  c10::musa::MUSAGuard device_guard(boxes.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  int boxes_num = boxes.size(0);

  const int col_blocks =
      (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  Tensor mask =
      at::empty({boxes_num, col_blocks}, boxes.options().dtype(at::kLong));

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(),
      (unsigned long long*)mask.data_ptr<int64_t>());

  at::Tensor keep_t = at::zeros(
      {boxes_num}, boxes.options().dtype(at::kBool).device(::at::musa::kMUSA));
  gather_keep_from_mask<<<1, min(col_blocks, THREADS_PER_BLOCK),
                          col_blocks * sizeof(unsigned long long), stream>>>(
      keep_t.data_ptr<bool>(), (unsigned long long*)mask.data_ptr<int64_t>(),
      boxes_num);

  auto keep_data = keep_t.nonzero().index({Slice(), 0});
  keep_num.fill_(at::Scalar(keep_data.size(0)));
  keep.index_put_({Slice(0, keep_data.size(0))}, keep_data);
  AT_MUSA_CHECK(musaGetLastError());
}