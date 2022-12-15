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
#include <cub/cub.cuh>

namespace cubit {

  enum { block_size = 1024 };

  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }
  
  template<typename key_t>
  inline static __device__ void putInOrder(key_t *const __restrict__ keys,
                                           uint32_t N,
                                           uint32_t a,
                                           uint32_t b)
  {
    if (b >= N) return;
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_a > key_b) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }
    
  template<typename key_t>
  inline static __device__ void putInOrder(key_t *const __restrict__ keys,
                                           uint64_t N,
                                           uint64_t a,
                                           uint64_t b)
  {
    if (b >= N) return;
    key_t key_a = keys[a];
    key_t key_b = keys[b];
    if (key_a > key_b) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
  }

  template<typename key_t>
  inline static __device__ void sort(key_t *const __restrict__ keys,
                                     uint32_t a,
                                     uint32_t b)
  {
    key_t key_a = keys[a];
    key_t key_b = keys[b];
#if 1
    keys[a] = (key_a<key_b)?key_a:key_b;
    keys[b] = (key_a<key_b)?key_b:key_a;
#else
    if (key_b < key_a) {
      keys[a] = key_b;
      keys[b] = key_a;
    }
#endif
  }


  template<typename key_t>
  __global__ void block_sort_up(key_t *const __restrict__ g_keys, uint32_t _N,
                                key_t *dbg, int dbg_a, int dbg_b)
  {
    if (dbg_a == 0 && dbg_b == 0) {
      dbg[threadIdx.x] = 0xfff;
      dbg[threadIdx.x+1024] = 0xfff;
      __syncthreads();
    }
    
    __shared__ key_t keys[2*1024];
    uint32_t blockStart = blockIdx.x*(2*1024);
    if (blockStart+threadIdx.x < _N)
      keys[threadIdx.x] = g_keys[blockStart+threadIdx.x];
    else
      keys[threadIdx.x] = 0xfff;
    if (1024+blockStart+threadIdx.x < _N)
      keys[1024+threadIdx.x] = g_keys[1024+blockStart+threadIdx.x];
    else
      keys[1024+threadIdx.x] = 0xfff;
    __syncthreads();
    
    // uint32_t N = _N - blockStart;

#if 1
#define DBG_SAVE(sa,sb)                           \
    if (sa==dbg_a && sb==dbg_b) {               \
      __syncthreads();                          \
      dbg[threadIdx.x] = keys[threadIdx.x];             \
      dbg[threadIdx.x+1024] = keys[threadIdx.x+1024];   \
    }
#else
#define DBG_SAVE(sa,sb)                           /* nothing */
#endif

    DBG_SAVE(0,0);

    int l, r, s;
    // ======== seq size 1 ==========
    {
      l = threadIdx.x+threadIdx.x;
      r = l + 1;

      if (dbg_a == 1 && dbg_b == 0 && r < _N)
        printf("(1.1)comparing %2i:%2i (values %i %i)\n",
               l,r,keys[l],keys[r]);
      // if (r < N)
      sort(keys,l,r);
    }

    DBG_SAVE(1,0);

    // ======== seq size 2 ==========
    {
      s    = (int)threadIdx.x & -2;
      l    = threadIdx.x+s;
      r    = l ^ (4-1);
      // if (r < N)
      // printf("(2.2)comparing %2i:%2i (values %i %i)\n",
      //        l,r,keys[l],keys[r]);
      // if (r < N)
      sort(keys,l,r);
      
      DBG_SAVE(2,1);
      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      // if (r < N)
      sort(keys,l,r);
      DBG_SAVE(2,0);
    }

    // ======== seq size 4 ==========
    {
      s    = (int)threadIdx.x & -4;
      l    = threadIdx.x+s;
      r    = l ^ (8-1);
      sort(keys,l,r);
      DBG_SAVE(4,2);
      
      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);
      DBG_SAVE(4,1);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
      DBG_SAVE(4,0);
    }

    // ======== seq size 8 ==========
    {
      s    = (int)threadIdx.x & -8;
      l    = threadIdx.x+s;
      r    = l ^ (16-1);
      sort(keys,l,r);
      
      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 16 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -16;
      l    = threadIdx.x+s;
      r    = l ^ (32-1);
      sort(keys,l,r);
      
      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 32 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -32;
      l    = threadIdx.x+s;
      r    = l ^ (64-1);
      sort(keys,l,r);
      
      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 64 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -64;
      l    = threadIdx.x+s;
      r    = l ^ (128-1);
      sort(keys,l,r);
      
      // ------ down seq size 32 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      sort(keys,l,r);

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 128 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -128;
      l    = threadIdx.x+s;
      r    = l ^ (256-1);
      sort(keys,l,r);
      
      // ------ down seq size 64 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      sort(keys,l,r);

      // ------ down seq size 32 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      sort(keys,l,r);

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 256 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -256;
      l    = threadIdx.x+s;
      r    = l ^ (512-1);
      sort(keys,l,r);
      
      // ------ down seq size 128 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      sort(keys,l,r);

      // ------ down seq size 64 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      sort(keys,l,r);

      // ------ down seq size 32 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      sort(keys,l,r);

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 512 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -512;
      l    = threadIdx.x+s;
      r    = l ^ (1024-1);
      sort(keys,l,r);
      
      // ------ down seq size 256 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      sort(keys,l,r);
      
      // ------ down seq size 128 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      sort(keys,l,r);

      // ------ down seq size 64 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      sort(keys,l,r);

      // ------ down seq size 32 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      sort(keys,l,r);

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    // ======== seq size 1024 ==========
    {
      __syncthreads();
      s    = (int)threadIdx.x & -1024;
      l    = threadIdx.x+s;
      r    = l ^ (2048-1);
      sort(keys,l,r);
      
      // ------ down seq size 512 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -512);
      r = l + 512;
      sort(keys,l,r);
      
      // ------ down seq size 256 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -256);
      r = l + 256;
      sort(keys,l,r);
      
      // ------ down seq size 128 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -128);
      r = l + 128;
      sort(keys,l,r);

      // ------ down seq size 64 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -64);
      r = l + 64;
      sort(keys,l,r);

      // ------ down seq size 32 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -32);
      r = l + 32;
      sort(keys,l,r);

      // ------ down seq size 16 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -16);
      r = l + 16;
      sort(keys,l,r);

      // ------ down seq size 8 ---------
      __syncthreads();
      l = threadIdx.x+((int)threadIdx.x & -8);
      r = l + 8;
      sort(keys,l,r);

      // ------ down seq size 4 ---------
      l = threadIdx.x+((int)threadIdx.x & -4);
      r = l + 4;
      sort(keys,l,r);

      // ------ down seq size 2 ---------
      l = threadIdx.x+((int)threadIdx.x & -2);
      r = l + 2;
      sort(keys,l,r);

      // ------ down seq size 1 ---------
      l = threadIdx.x+threadIdx.x;
      r = l + 1;
      sort(keys,l,r);
    }

    __syncthreads();
    if (blockStart+threadIdx.x < _N) g_keys[blockStart+threadIdx.x] = keys[threadIdx.x];
    if (1024+blockStart+threadIdx.x < _N) g_keys[1024+blockStart+threadIdx.x] = keys[1024+threadIdx.x];
  }

  
  template<typename key_t, bool reverse>
  __global__ void d_bitonic(key_t *const __restrict__ keys, uint32_t N, int logSegLen)
  {
    const uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= N) return;
    if (reverse) {
      const uint32_t seg  = tid >> logSegLen;
      
      const uint32_t segBegin = (seg << (logSegLen+1));
      const uint32_t segLen   = 1<<logSegLen;
    
      const uint32_t lane = cub::BFE(tid,0,logSegLen);
    
      const uint32_t l = segBegin + lane;
      const uint32_t r = (segBegin + (2*segLen-1) - lane);
      putInOrder(keys,N,l,r);
    } else {
      const uint32_t lane = cub::BFE(tid,0,logSegLen);
      const uint32_t seg  = tid >> logSegLen;
      
      const uint32_t segBegin = (seg << (logSegLen+1));
      const uint32_t segLen   = 1<<logSegLen;
    
      const uint32_t l = segBegin + lane;
      const uint32_t r = (l+segLen);
      putInOrder(keys,N,l,r);
    }
  }


  template<typename key_t, bool first_reverse>
  __global__ void d_bitonic_block(key_t *const __restrict__ _keys, uint32_t N, uint32_t logSegLen)
  {
    if (N <= blockIdx.x*(2*cubit::block_size)) return;
    N -= blockIdx.x*(2*cubit::block_size);

    uint32_t segLen = 1<<logSegLen;
    __shared__ key_t l_keys[2*cubit::block_size];
    if (threadIdx.x < N)
      l_keys[threadIdx.x] = _keys[blockIdx.x*(2*cubit::block_size)+threadIdx.x];
    if (threadIdx.x+cubit::block_size < N)
      l_keys[threadIdx.x+cubit::block_size] =
        _keys[blockIdx.x*(2*cubit::block_size)+(threadIdx.x+cubit::block_size)];
    __syncthreads();
    key_t *const __restrict__ keys = l_keys;
    uint32_t seg  = threadIdx.x >> logSegLen;
      
    uint32_t segBegin = (seg << (logSegLen+1));
    uint32_t segBeginHalf = seg << logSegLen;
    // uint32_t segLen   = 1<<logSegLen;
    
    // uint32_t lane = threadIdx.x & (segLen-1);
    const uint32_t lane = cub::BFE(threadIdx.x,0,logSegLen);
    
    // uint32_t l = segBegin + lane;
    uint32_t l = segBeginHalf+segBeginHalf+lane;
    uint32_t r
      = first_reverse
      ? (segBegin + (2*segLen-1) - lane)
      : (l+segLen);
    
    putInOrder(keys,N,l,r);

#pragma unroll(5)
    while (logSegLen > 0) {
      if (logSegLen >= 5) __syncthreads();
      
      --logSegLen;
      uint32_t seg  = threadIdx.x >> logSegLen;
      
      uint32_t segBegin = (seg << (logSegLen+1));
      uint32_t segLen   = 1<<logSegLen;
      uint32_t lane = threadIdx.x & (segLen-1);
      
      uint32_t l = segBegin + lane;
      uint32_t r = (l+segLen);
      
      putInOrder(keys,N,l,r);
    }
    __syncthreads();
    if (threadIdx.x < N)
      _keys[blockIdx.x*(2*cubit::block_size)+threadIdx.x]
        = l_keys[threadIdx.x];
    if (threadIdx.x+cubit::block_size < N)
      _keys[blockIdx.x*(2*cubit::block_size)+(threadIdx.x+cubit::block_size)]
        = l_keys[threadIdx.x+cubit::block_size];
  }
  
  template<typename key_t>
  inline void sort(key_t *const __restrict__ d_values,
                   size_t numValues,
                   key_t *dbg=0,
                   int dbg_a=-1,
                   int dbg_b=-1,
                   cudaStream_t stream=0)

  {
#if 1
    int bs = 1024;
    int numValuesPerBlock = 2*bs;
    int nb = divRoundUp((int)numValues,numValuesPerBlock);
    block_sort_up<<<nb,bs>>>(d_values,numValues,dbg,dbg_a,dbg_b);
    CUBIT_CUDA_SYNC_CHECK(); fflush(0);
#else
    int N = numValues;
    int bs = cubit::block_size;
    int nb = (N+bs-1)/bs;
    for (int logLen = 0; (1<<logLen) <= N; logLen++) {
      if ((1<<logLen) <= bs)
        d_bitonic_block<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
      else {
        d_bitonic<key_t,true><<<nb,bs,0,stream>>>(d_values,N,logLen);
        for (int backLen = logLen; backLen > 0; --backLen) {
          if ((1<<(backLen-1)) <= bs) {
            d_bitonic_block<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
            break;
          }
          d_bitonic<key_t,false><<<nb,bs,0,stream>>>(d_values,N,backLen-1);
        }
      }
    }
#endif
  }
  
}

