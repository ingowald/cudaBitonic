#include "../cubit/cubit.h"
#include <cub/cub.cuh>
#include <vector>

using namespace cubit::common;

inline bool sorted(const std::vector<int> &values)
{
  for (int i=1;i<values.size();i++)
    if (values[i] < values[i-1]) return false;
  return true;
}

template<typename T>
void print(T *values, size_t N, int S) {
  int eol = N;
  while (values[eol-1] == INT_MAX) --eol;
  for (int i=0;i<eol;i++) {
    if (i%S == 0)
      printf("(");
    else 
      printf(" ");
    if (values[i] == INT_MAX)
      printf("---");
    else
      printf("%3i",values[i]);
    if (i%S == (S-1) || i==(N-1))
      printf(")");
    else 
      printf(" ");
  }
  printf("\n\n");
}

template<typename T>
void checkBlocks(T *values, size_t N, int S)
{
  std::cout << "### checking blocks of size " << S << std::endl;
  for (int i=0;i<N;i++) {
    if ((i % S) == 0)
      continue;

    if (values[i] < values[i-1]) {
      printf("values[%i] = %i vs [%i] = %i\n",
             i,values[i],i-1,values[i-1]);
      throw std::runtime_error("not sorted at position "+std::to_string(i));
    }
  }
  std::cout << "... OK" << std::endl;
}

template<typename T>
void test_keys(const std::vector<T> &h_values)
{
  T *d_values = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_values,h_values.size()*sizeof(T)));
  if (!d_values) throw std::runtime_error("could not malloc...");
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));

  T *d_bitonic = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_bitonic,h_values.size()*sizeof(T)));
  if (!d_bitonic) throw std::runtime_error("could not malloc...");

  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  T *d_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_in,h_values.size()*sizeof(T)));
  if (!d_cub_radix_in) throw std::runtime_error("could not malloc...");
  T *d_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_out,h_values.size()*sizeof(T)));
  if (!d_cub_radix_out) throw std::runtime_error("could not malloc...");
  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                 d_cub_radix_in,d_cub_radix_out,h_values.size());
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_tmp,cub_radix_tmp_size));
  if (!d_cub_radix_tmp) throw std::runtime_error("could not malloc...");
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
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<T> h_results(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_cub_radix_out,h_values.size()*sizeof(T),cudaMemcpyDefault));
  // CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_bitonic,h_values.size()*sizeof(T),cudaMemcpyDefault));
  if (!sorted(h_results))
    throw std::runtime_error("not sorted...");

  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix) << "  vs bitonic " << prettyDouble(t_bitonic) << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)" << std::endl;


  CUBIT_CUDA_CALL(Free(d_values));
  CUBIT_CUDA_CALL(Free(d_bitonic));
  CUBIT_CUDA_CALL(Free(d_cub_radix_in));
  CUBIT_CUDA_CALL(Free(d_cub_radix_out));
  CUBIT_CUDA_CALL(Free(d_cub_radix_tmp));
}

template<typename T>
void test_pairs(const std::vector<T> &h_values)
{
  T *d_keys = 0;
  T *d_values = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_values,h_values.size()*sizeof(T)));
  if (!d_values) throw std::runtime_error("could not malloc...");
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));

  CUBIT_CUDA_CALL(Malloc((void**)&d_keys,h_values.size()*sizeof(T)));
  if (!d_keys) throw std::runtime_error("could not malloc...");
  CUBIT_CUDA_CALL(Memcpy(d_keys,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));
  
  T *keys_bitonic = 0;
  T *vals_bitonic = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_bitonic,h_values.size()*sizeof(T)));
  if (!keys_bitonic) throw std::runtime_error("could not malloc...");
  CUBIT_CUDA_CALL(Malloc((void**)&vals_bitonic,h_values.size()*sizeof(T)));
  if (!vals_bitonic) throw std::runtime_error("could not malloc...");

  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  T *keys_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_cub_radix_in,h_values.size()*sizeof(T)));
  if (!keys_cub_radix_in) throw std::runtime_error("could not malloc...");
  T *keys_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&keys_cub_radix_out,h_values.size()*sizeof(T)));
  if (!keys_cub_radix_out) throw std::runtime_error("could not malloc...");

  T *vals_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&vals_cub_radix_in,h_values.size()*sizeof(T)));
  if (!vals_cub_radix_in) throw std::runtime_error("could not malloc...");
  T *vals_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&vals_cub_radix_out,h_values.size()*sizeof(T)));
  if (!vals_cub_radix_out) throw std::runtime_error("could not malloc...");

  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortPairs(d_cub_radix_tmp,cub_radix_tmp_size,
                                  keys_cub_radix_in,keys_cub_radix_out,
                                  vals_cub_radix_in,vals_cub_radix_out,
                                  h_values.size());
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_tmp,cub_radix_tmp_size));
  if (!d_cub_radix_tmp) throw std::runtime_error("could not malloc...");
  int nRepeats = 100;
  
  // ------------------------------------------------------------------
  // CUB ***EXEC***
  // ------------------------------------------------------------------
  double t0_cub_radix = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(vals_cub_radix_in,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    CUBIT_CUDA_CALL(Memcpy(keys_cub_radix_in,d_keys,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cub::DeviceRadixSort::SortPairs(d_cub_radix_tmp,cub_radix_tmp_size,
                                    keys_cub_radix_in,keys_cub_radix_out,
                                    vals_cub_radix_in,vals_cub_radix_out,
                                   h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_cub_radix = (getCurrentTime() - t0_cub_radix)/nRepeats;
  
  // ------------------------------------------------------------------
  // BITONIC
  // ------------------------------------------------------------------
  double t0_bitonic = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(keys_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    CUBIT_CUDA_CALL(Memcpy(vals_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(keys_bitonic,vals_bitonic,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<T> h_results(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_results.data(),vals_bitonic,h_values.size()*sizeof(T),cudaMemcpyDefault));
  if (!sorted(h_results))
    throw std::runtime_error("not sorted...");

  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix) << "  vs bitonic " << prettyDouble(t_bitonic) << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)" << std::endl;


  CUBIT_CUDA_CALL(Free(d_values));
  CUBIT_CUDA_CALL(Free(vals_bitonic));
  CUBIT_CUDA_CALL(Free(vals_cub_radix_in));
  CUBIT_CUDA_CALL(Free(vals_cub_radix_out));
  CUBIT_CUDA_CALL(Free(d_cub_radix_tmp));
  CUBIT_CUDA_CALL(Free(keys_bitonic));
  CUBIT_CUDA_CALL(Free(keys_cub_radix_in));
  CUBIT_CUDA_CALL(Free(keys_cub_radix_out));
}

int main(int ac, char **av)
{
  while (true) {
    // int N = 7654;
    int N = 50*1000000;//1+int(powf(2.f,24*sqrtf(drand48())));
    std::vector<int> values(N);
    for (int i=0;i<N;i++) {
      values[i] = (random() % (1<<30) - 128000);
    }

    std::cout << "testing array of size " << N << std::endl;
    test_keys(values);
    // test_pairs(values);
  }
}

