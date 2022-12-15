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
  while (values[eol-1] == 0xfff) --eol;
  for (int i=0;i<eol;i++) {
    if (i%S == 0)
      printf("(");
    else 
      printf(" ");
    if (values[i] == 0xfff)
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
  for (int i=0;i<N;i++) {
    if ((i % 1024) == 0)
      continue;

    if (values[i] < values[i-1]) {
      printf("values[%i] = %i vs [%i] = %i\n",
             i,values[i],i-1,values[i-1]);
      throw std::runtime_error("not sorted at position "+std::to_string(i));
    }
  }
}

template<typename T>
void testIt(const std::vector<T> &h_values)
{
  T *d_values = 0;
  CUBIT_CUDA_CALL(MallocManaged((void**)&d_values,h_values.size()*sizeof(T)));
  if (!d_values) throw std::runtime_error("could not malloc...");
  CUBIT_CUDA_CALL(Memcpy(d_values,h_values.data(),h_values.size()*sizeof(T),cudaMemcpyDefault));

  T *d_bitonic = 0;
  CUBIT_CUDA_CALL(MallocManaged((void**)&d_bitonic,h_values.size()*sizeof(T)));
  if (!d_bitonic) throw std::runtime_error("could not malloc...");

  // ------------------------------------------------------------------
  // CUB ***INIT***
  // ------------------------------------------------------------------
  T *d_cub_radix_in = 0;
  CUBIT_CUDA_CALL(MallocManaged((void**)&d_cub_radix_in,h_values.size()*sizeof(T)));
  if (!d_cub_radix_in) throw std::runtime_error("could not malloc...");
  T *d_cub_radix_out = 0;
  CUBIT_CUDA_CALL(MallocManaged((void**)&d_cub_radix_out,h_values.size()*sizeof(T)));
  if (!d_cub_radix_out) throw std::runtime_error("could not malloc...");
  void *d_cub_radix_tmp = 0;
  size_t  cub_radix_tmp_size = 0;
  cub::DeviceRadixSort::SortKeys(d_cub_radix_tmp,cub_radix_tmp_size,
                                  d_cub_radix_in,d_cub_radix_out,h_values.size());
  CUBIT_CUDA_CALL(MallocManaged((void**)&d_cub_radix_tmp,cub_radix_tmp_size));
  if (!d_cub_radix_tmp) throw std::runtime_error("could not malloc...");
  int nRepeats = 100;
  
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
#if 1
  {
    key_t *dbg;
    cudaMallocManaged((void**)&dbg,1024*sizeof(key_t));
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    print(d_bitonic,h_values.size(),1); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,0,0);
    std::cout << "0:0: " << std::endl;
    print(dbg,1024,1); std::cout << "\n\n";

    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,1,0);
    std::cout << "1:0: " << std::endl;
    print(dbg,1024,2); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,2,1);
    std::cout << "2:1: " << std::endl;
    print(dbg,1024,4); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,2,0);
    std::cout << "2:0: " << std::endl;
    print(dbg,1024,4); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,4,2);
    std::cout << "4:2: " << std::endl;
    print(dbg,1024,8); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,4,1);
    std::cout << "4:1: " << std::endl;
    print(dbg,1024,8); std::cout << "\n\n";
    
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,4,0);
    std::cout << "4:0: " << std::endl;
    print(dbg,1024,8); std::cout << "\n\n";



    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size(),dbg,-1,-1);
    
    
    CUBIT_CUDA_SYNC_CHECK();
    print(d_bitonic,h_values.size(),1024);
    checkBlocks(d_bitonic,h_values.size(),1024);
    // print(d_bitonic,h_values.size(),8);
    exit(0);
  }
#endif
  double t0_bitonic = getCurrentTime();
  for (int i=0;i<nRepeats;i++) {
    CUBIT_CUDA_CALL(Memcpy(d_bitonic,d_values,h_values.size()*sizeof(T),cudaMemcpyDefault));
    cubit::sort(d_bitonic,h_values.size());
  }
  CUBIT_CUDA_SYNC_CHECK();
  double t_bitonic = (getCurrentTime() - t0_bitonic)/nRepeats;
  
  CUBIT_CUDA_SYNC_CHECK();

  std::vector<T> h_results(h_values.size());
  CUBIT_CUDA_CALL(Memcpy(h_results.data(),d_bitonic,h_values.size()*sizeof(T),cudaMemcpyDefault));
  if (!sorted(h_results))
    throw std::runtime_error("not sorted...");

  std::cout << "time(s) : cub radix = " << prettyDouble(t_cub_radix) << "  vs bitonic " << prettyDouble(t_bitonic) << " (that's " << (t_bitonic/t_cub_radix) << "x faster than us)" << std::endl;


  CUBIT_CUDA_CALL(Free(d_values));
  CUBIT_CUDA_CALL(Free(d_bitonic));
  CUBIT_CUDA_CALL(Free(d_cub_radix_in));
  CUBIT_CUDA_CALL(Free(d_cub_radix_out));
  CUBIT_CUDA_CALL(Free(d_cub_radix_tmp));
}

int main(int ac, char **av)
{
  while (true) {
    int N = 7654;
    // int N = 50*1000000;//1+int(powf(2.f,24*sqrtf(drand48())));
    std::vector<int> values(N);
    for (int i=0;i<N;i++) {
      values[i] = (random() % 1000);
    }
      // values[i] = (random() % N) - N/4;

    std::cout << "testing array of size " << N << std::endl;
    testIt(values);
  }
}

