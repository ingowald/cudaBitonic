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

#include "../cubit/cubit.h"
#include <cub/cub.cuh>
#include <vector>
#include <random>
#include <algorithm>
#include <typeinfo>

using namespace cubit::common;

template<typename T>
inline bool sorted(const std::vector<T> &values)
{
  for (size_t i=1;i<values.size();i++)
    if (values[i] < values[i-1]) return false;
  return true;
}

template<typename T>
inline bool asExpected(const std::vector<T> &ours_sorted,
                       const std::vector<T> &input_unsorted)
{
  std::vector<T> reference_sorted = input_unsorted;
  std::sort(reference_sorted.begin(),reference_sorted.end());
  return reference_sorted == ours_sorted;
}

template<typename T>
inline bool allValuesAccountedFor(std::vector<T> vals0,
                                  std::vector<T> vals1)
{
  std::sort(vals0.begin(),vals0.end());
  std::sort(vals1.begin(),vals1.end());
  return vals0 == vals1;
}

template<typename T>
void test_keys(const std::vector<T> &h_values)
{
  // upload test data to the device
  T *d_values = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_values,h_values.size()*sizeof(T)));
  if (!d_values) throw std::runtime_error("could not malloc (1)...");
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));
  
  // create a copy of the uploaded data, so we can keep re-sorting the
  // same array multiple timing for timing measrements
  T *d_bitonic = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_bitonic,h_values.size()*sizeof(T)));
  
  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  T *d_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_in,h_values.size()*sizeof(T)));
  if (!d_cub_radix_in) throw std::runtime_error("could not malloc (2)...");
  T *d_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_out,h_values.size()*sizeof(T)));
  if (!d_cub_radix_out) throw std::runtime_error("could not malloc (3)...");
  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                 d_cub_radix_in,d_cub_radix_out,h_values.size());
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_tmp,cub_radix_tmp_size));
  if (!d_cub_radix_tmp) throw std::runtime_error("could not malloc (4)...");
  int nRepeats = 10;
  
  // ------------------------------------------------------------------
  // CUB ***EXEC***
  // ------------------------------------------------------------------
  double t0_cub_radix = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(d_cub_radix_in,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                   d_cub_radix_in,d_cub_radix_out,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_cub_radix = (getCurrentTime() - t0_cub_radix)/nRepeats;
  
  // ------------------------------------------------------------------
  // BITONIC
  // ------------------------------------------------------------------
  double t0_bitonic = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    // re-set data array to input values
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    // and (re-)sort
    cubit::sort(d_bitonic,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<T> h_results(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_cub_radix_out,h_values.size()*sizeof(T),cudaMemcpyDefault));
  // CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_bitonic,h_values.size()*sizeof(T),cudaMemcpyDefault));
  if (!sorted(h_results) || !asExpected(h_results,h_values)) {
    std::cout << CUBIT_TERMINAL_RED << "*** TEST FAILED ***" << CUBIT_TERMINAL_DEFAULT << std::endl;
    throw std::runtime_error("not sorted...");
  } else
    std::cout << CUBIT_TERMINAL_GREEN << "... ok." << CUBIT_TERMINAL_DEFAULT << std::endl;

  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix)
            << " vs bitonic " << prettyDouble(t_bitonic)
            << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)"
            << std::endl;

  CUBIT_CUDA_CALL(Free(d_values));
  CUBIT_CUDA_CALL(Free(d_bitonic));
}

template<typename KeyT, typename ValueT>
void test_pairs(const std::vector<KeyT> &h_keys,
                const std::vector<ValueT> &h_values)
{
  KeyT *d_keys = 0;
  ValueT *d_values = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_values,h_values.size()*sizeof(ValueT)));
  if (!d_values) throw std::runtime_error("could not malloc (5)...");
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(ValueT),cudaMemcpyDefault));

  CUBIT_CUDA_CALL(Malloc((void**)&d_keys,h_keys.size()*sizeof(KeyT)));
  if (!d_keys) throw std::runtime_error("could not malloc (6)...");
  CUBIT_CUDA_CALL(Memcpy(d_keys,h_keys.data(),h_keys.size()*sizeof(KeyT),cudaMemcpyDefault));
  
  KeyT *keys_bitonic = 0;
  ValueT *values_bitonic = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_bitonic,h_keys.size()*sizeof(KeyT)));
  if (!keys_bitonic) throw std::runtime_error("could not malloc (7)...");
  CUBIT_CUDA_CALL(Malloc((void**)&values_bitonic,h_values.size()*sizeof(ValueT)));
  if (!values_bitonic) throw std::runtime_error("could not malloc (8)...");

  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  KeyT *keys_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_cub_radix_in,h_keys.size()*sizeof(KeyT)));
  if (!keys_cub_radix_in) throw std::runtime_error("could not malloc (9)...");
  KeyT *keys_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_cub_radix_out,h_keys.size()*sizeof(KeyT)));
  if (!keys_cub_radix_out) throw std::runtime_error("could not malloc (10)...");

  ValueT *values_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&values_cub_radix_in,h_values.size()*sizeof(ValueT)));
  if (!values_cub_radix_in) throw std::runtime_error("could not malloc (11)...");
  ValueT *values_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&values_cub_radix_out,h_values.size()*sizeof(ValueT)));
  if (!values_cub_radix_out) throw std::runtime_error("could not malloc (12)...");

  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortPairs(d_cub_radix_tmp,cub_radix_tmp_size,
                                  keys_cub_radix_in,keys_cub_radix_out,
                                  values_cub_radix_in,values_cub_radix_out,
                                  h_values.size());
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_tmp,cub_radix_tmp_size));
  if (cub_radix_tmp_size && !d_cub_radix_tmp) throw std::runtime_error("could not malloc (13)...");
  int nRepeats = 100;
  
  // ------------------------------------------------------------------
  // CUB ***EXEC***
  // ------------------------------------------------------------------
  double t0_cub_radix = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(keys_cub_radix_in,d_keys,h_keys.size()*sizeof(KeyT),cudaMemcpyDefault));
    CUBIT_CUDA_CALL(Memcpy(values_cub_radix_in,d_values,h_values.size()*sizeof(ValueT),cudaMemcpyDefault));
    cub::DeviceRadixSort::SortPairs(d_cub_radix_tmp,cub_radix_tmp_size,
                                    keys_cub_radix_in,keys_cub_radix_out,
                                    values_cub_radix_in,values_cub_radix_out,
                                    h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_cub_radix = (getCurrentTime() - t0_cub_radix)/nRepeats;
  
  // ------------------------------------------------------------------
  // BITONIC
  // ------------------------------------------------------------------
  double t0_bitonic = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(keys_bitonic,d_keys,h_keys.size()*sizeof(KeyT),cudaMemcpyDefault));
    CUBIT_CUDA_CALL(Memcpy(values_bitonic,d_values,h_values.size()*sizeof(ValueT),cudaMemcpyDefault));
    cubit::sort(keys_bitonic,values_bitonic,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<KeyT> h_result_keys(h_keys.size());
  CUBIT_CUDA_CALL(Memcpy(h_result_keys.data(),keys_bitonic,
                         h_keys.size()*sizeof(KeyT),cudaMemcpyDefault));
  std::vector<ValueT> h_result_values(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_result_values.data(),values_bitonic,
                         h_values.size()*sizeof(ValueT),cudaMemcpyDefault));
  if (!sorted(h_result_keys) ||
      !asExpected(h_result_keys,h_keys) ||
      !allValuesAccountedFor(h_result_values,h_values))
    {
      std::cout << CUBIT_TERMINAL_RED << "*** TEST FAILED ***" << CUBIT_TERMINAL_DEFAULT
                << std::endl;
      throw std::runtime_error("not sorted...");
    }
  else
    std::cout << CUBIT_TERMINAL_GREEN << "... ok." << CUBIT_TERMINAL_DEFAULT << std::endl;
  
  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix)
            << " vs bitonic " << prettyDouble(t_bitonic)
            << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)"
            << std::endl;

  CUBIT_CUDA_CALL(Free(d_values));
  CUBIT_CUDA_CALL(Free(values_bitonic));
  CUBIT_CUDA_CALL(Free(values_cub_radix_in));
  CUBIT_CUDA_CALL(Free(values_cub_radix_out));
  CUBIT_CUDA_CALL(Free(d_cub_radix_tmp));
  CUBIT_CUDA_CALL(Free(keys_bitonic));
  CUBIT_CUDA_CALL(Free(keys_cub_radix_in));
  CUBIT_CUDA_CALL(Free(keys_cub_radix_out));
}


template<typename T> struct Random;

// define these dummy types to make the macros work on 'uint32' instead of 'uint32_t'
typedef int32_t int32;
typedef uint32_t uint32;
typedef uint64_t uint64;

template<> struct Random<float>
{
  Random(std::mt19937 &gen) : gen(gen), dis(0.f,1.f) {}
  
  inline float operator()() { return dis(gen); }
  
  std::mt19937 &gen;
  std::uniform_real_distribution<float> dis;
};

template<> struct Random<double>
{
  Random(std::mt19937 &gen) : gen(gen), dis(0.f,1.f) {}
  
  inline double operator()() { return dis(gen); }
  
  std::mt19937 &gen;
  std::uniform_real_distribution<double> dis;
};

template<> struct Random<uint64_t>
{
  Random(std::mt19937 &gen) : gen(gen) {}
  
  inline uint64_t operator()() {
    uint32_t lo = (uint32_t)(gen());
    uint32_t hi = (uint32_t)(gen());
    return lo | uint64_t(hi)<<32;
  }
  
  std::mt19937 &gen;
};

template<> struct Random<uint32_t>
{
  Random(std::mt19937 &gen) : gen(gen) {}
  
  inline uint32_t operator()() { return (uint32_t)(gen()); }
  
  std::mt19937 &gen;
};

template<> struct Random<int32_t>
{
  Random(std::mt19937 &gen) : gen(gen) {}
  
  inline int32_t operator()() { return (int32_t)(gen()); }
  
  std::mt19937 &gen;
};

int main(int ac, char **av)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  Random<float> sizeRandom(gen);
  for (int rep=0;rep<1000;rep++) {
    int N = int(powf(2.f,28*sqrtf(sizeRandom())));
    if (ac > 1)
      N = std::stoi(av[1]);
#ifdef VALUE_T
    std::cout << CUBIT_TERMINAL_BLUE
              << "testing sorting on " << prettyNumber(N)
              << " keys:value pairs of key type " << typeid(KEY_T).name()
              << " and value type " << typeid(VALUE_T).name()
              << CUBIT_TERMINAL_DEFAULT << std::endl;
#else
    std::cout << CUBIT_TERMINAL_BLUE
              << "testing sorting on " << prettyNumber(N)
              << " keys of key type " << typeid(KEY_T).name()
              << CUBIT_TERMINAL_DEFAULT << std::endl;
#endif
    static Random<KEY_T> key_random(gen);
    std::vector<KEY_T> keys(N);
    for (int i=0;i<N;i++) keys[i] = key_random();
#ifdef VALUE_T
    std::cout << "testing sorting on " << prettyNumber(N) << " values..." << std::endl;
    static Random<VALUE_T> value_random(gen);
    std::vector<VALUE_T> values(N);
    for (int i=0;i<N;i++) {
      values[i] = value_random();
    }
#endif

#ifdef VALUE_T
    test_pairs(keys,values);
#else
    test_keys(keys);
#endif
  }
}

