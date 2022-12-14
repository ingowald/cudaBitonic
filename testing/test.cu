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
void testIt(const std::vector<T> &h_values)
{
  T *d_values = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_values,h_values.size()*sizeof(T)));
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));

  T *d_bitonic = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_bitonic,h_values.size()*sizeof(T)));

  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  T *d_cub_radix_in = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_in,h_values.size()*sizeof(T)));
  T *d_cub_radix_out = 0;
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_out,h_values.size()*sizeof(T)));
  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                  d_cub_radix_in,d_cub_radix_out,h_values.size());
  CUBIT_CUDA_CALL(Malloc((void**)&d_cub_radix_tmp,cub_radix_tmp_size));

  int nRepeats = 1000;
  
  // ------------------------------------------------------------------
  // CUB ***EXEC***
  // ------------------------------------------------------------------
  double t0_cub_radix = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(d_cub_radix_in,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                   d_cub_radix_in,d_cub_radix_out,h_values.size());
  }
  double t_cub_radix = (getCurrentTime() - t0_cub_radix)/nRepeats;
  
  // ------------------------------------------------------------------
  // BITONIC
  // ------------------------------------------------------------------
  double t0_bitonic = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size());
  }
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<T> h_results(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_bitonic,h_values.size()*sizeof(T),cudaMemcpyDefault));
  if (!sorted(h_results))
    throw std::runtime_error("not sorted...");

  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix) << "  vs bitonic " << prettyDouble(t_bitonic) << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)" << std::endl;
}

int main(int ac, char **av)
{
  while (true) {
    int N = 1+int(powf(2.f,20*drand48()));
    std::vector<int> values(N);
    for (int i=0;i<N;i++)
      values[i] = (random() % N) - N/4;

    std::cout << "testing array of size " << N << std::endl;
    testIt(values);
  }
}

