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
    
  template<typename key_t>
  inline void sort(key_t *d_values, size_t numValues, cudaStream_t stream=0)
  {
    int N = numValues;
    int bs = 128;
    int nb = (N+bs-1)/bs;
    for (int logLen = 0; (1<<logLen) <= N; logLen++) {
      d_bitonic<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
      for (int backLen = logLen; backLen > 0; --backLen) {
        d_bitonic<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
      }
    }
  }
  
}

