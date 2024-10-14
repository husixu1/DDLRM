#pragma once

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda_metrics/gpu_error.hpp"

struct l2flush
{
  __forceinline__ l2flush()
  {
    int dev_id{};
    GPU_ERROR(cudaGetDevice(&dev_id));
    GPU_ERROR(cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    if (m_l2_size > 0)
    {
      void *buffer = m_l2_buffer;
      GPU_ERROR(cudaMalloc(&buffer, m_l2_size));
      m_l2_buffer = reinterpret_cast<int *>(buffer);
    }
  }

  __forceinline__ ~l2flush()
  {
    if (m_l2_buffer)
    {
      GPU_ERROR(cudaFree(m_l2_buffer));
    }
  }

  __forceinline__ void flush()
  {
    if (m_l2_size > 0)
    {
      GPU_ERROR(cudaMemset(m_l2_buffer, 0, m_l2_size));
    }
  }

private:
  int m_l2_size{};
  int *m_l2_buffer{};
};