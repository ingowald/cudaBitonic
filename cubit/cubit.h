// ======================================================================== //
// Copyright 2022-2022 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <cuda_runtime.h>
#include "../cubit/common.h"

namespace cubit {

  enum { block_size = 1024 };
  
  template<typename key_t>
  inline static __device__ void putInOrder(key_t *keys, int N, int a, int b)
  {
    if (b >= N) return;
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_a > key_b) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }
    
  template<typename key_t, bool reverse>
  __global__ void d_bitonic(key_t *keys, int N, int logSegLen)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int seg  = tid >> logSegLen;
      
    int segBegin = (seg << (logSegLen+1));
    int segLen   = 1<<logSegLen;
    
    int lane = tid & (segLen-1);
    
    int l = segBegin + lane;
    int r
      = reverse
      ? (segBegin + (2*segLen-1) - lane)
      : (l+segLen);
    
    putInOrder(keys,N,l,r);
  }


#define USE_SHM 1

#if USE_SHM
  template<typename key_t, bool first_reverse>
  __global__ void d_bitonic_block(key_t *_keys, int N, int logSegLen)
  {
    // int tid = threadIdx.x + blockIdx.x*blockDim.x;
    N -= 2*blockIdx.x*blockDim.x;
    // if (blockIdx.x*blockDim.x >= N) return;

    int segLen = 1<<logSegLen;
    __shared__ key_t l_keys[2*cubit::block_size];
    if (threadIdx.x < N)
      l_keys[threadIdx.x] = _keys[2*blockIdx.x*blockDim.x+threadIdx.x];
    if (threadIdx.x+blockDim.x < N)
      l_keys[threadIdx.x+blockDim.x] = _keys[2*blockIdx.x*blockDim.x+threadIdx.x+blockDim.x];
    __syncthreads();
    key_t *keys = l_keys;
    int seg  = threadIdx.x >> logSegLen;
      
    int segBegin = (seg << (logSegLen+1));
    // int segLen   = 1<<logSegLen;
    
    int lane = threadIdx.x & (segLen-1);
    
    int l = segBegin + lane;
    int r
      = first_reverse
      ? (segBegin + (2*segLen-1) - lane)
      : (l+segLen);
    
    putInOrder(keys,N,l,r);

    while (logSegLen > 0) {
      __syncthreads();
      
      --logSegLen;
      int seg  = threadIdx.x >> logSegLen;
      
      int segBegin = (seg << (logSegLen+1));
      int segLen   = 1<<logSegLen;
      
      int lane = threadIdx.x & (segLen-1);
      
      int l = segBegin + lane;
      int r = (l+segLen);
      
      putInOrder(keys,N,l,r);
    }
    __syncthreads();
    if (threadIdx.x < N)
      _keys[2*blockIdx.x*blockDim.x+threadIdx.x] = l_keys[threadIdx.x];
    if (threadIdx.x+blockDim.x < N)
      _keys[2*blockIdx.x*blockDim.x+threadIdx.x+blockDim.x] = l_keys[threadIdx.x+blockDim.x];
  }
#else
  template<typename key_t, bool first_reverse>
  __global__ void d_bitonic_block(key_t *_keys, int N, int logSegLen)
  {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    key_t *keys = _keys;
    int seg  = tid >> logSegLen;
      
    int segBegin = (seg << (logSegLen+1));
    int segLen   = 1<<logSegLen;
    
    int lane = tid & (segLen-1);
    
    int l = segBegin + lane;
    int r
      = first_reverse
      ? (segBegin + (2*segLen-1) - lane)
      : (l+segLen);
    
    putInOrder(keys,N,l,r);

    while (logSegLen > 0) {
      __syncthreads();
      
      --logSegLen;
      int seg  = tid >> logSegLen;
      
      int segBegin = (seg << (logSegLen+1));
      int segLen   = 1<<logSegLen;
      
      int lane = tid & (segLen-1);
      
      int l = segBegin + lane;
      int r = (l+segLen);
      
      putInOrder(keys,N,l,r);
    }
  }
#endif    
  
  template<typename key_t>
  inline void sort(key_t *d_values, size_t numValues, cudaStream_t stream=0)
  {
    int N = numValues;
    int bs = cubit::block_size;//128;
    int nb = (N+bs-1)/bs;
#if 1
    for (int logLen = 0; (1<<logLen) <= N; logLen++) {
      if ((1<<logLen) <= bs)
        d_bitonic_block<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
      else {
        d_bitonic<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
        for (int backLen = logLen; backLen > 0; --backLen) {
          if ((2<<(backLen-1)) <= bs) {
            d_bitonic_block<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
            break;
          }
          d_bitonic<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
        }
      }
    }
#else
    for (int logLen = 0; (1<<logLen) <= N; logLen++) {
      d_bitonic<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
      for (int backLen = logLen; backLen > 0; --backLen) {
        d_bitonic<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
      }
    }
#endif
  }
  
}

