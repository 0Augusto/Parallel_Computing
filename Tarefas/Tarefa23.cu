/*
 * @author(s):  Henrique Augusto Rodrigues
 * Matricula:   XXXXXX
 * Tarefa 22
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa23.cu -o Tarefa23 -fopenmp   
 * Windows:     gcc -o Tarefa23.exe   -fopenmp
 * macbook:     gcc -o Tarefa23.cu  Tarefa23 -openmp
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa23   
 * Windows:          Tarefa23  
 * MacBook:   time ./Tarefa23
 */

/*
Não sei o motivo do erro a seguir, gcc8: error: Tarefa23: No such file or directory
gcc8: fatal error: no input files
Então, os tempos não são meus, porém, conversando com os veteranos que fizeram a disciplinas e me auxiliaram rodando o código (eles mesmos), 
os tempos encontrados estão ao final do código.
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

__global__ void Tarefa23_cuda(double* a, double* b, double* c, int width){
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   int row = blockIdx.y * blockDim.y + threadIdx.y;

   if(col < width && row < width){
      double sum = 0;

      for (int i=0; i<width; i++){
         double x = a[row * width + i];
         double y = b[i*width + col];
         sum += x*y;
      }
      c[row * width + col] = sum;
   }
}
int main()
{
  int width = 2000;
  double *a = (double*) malloc (width * width * sizeof(double));
  double *b = (double*) malloc (width * width * sizeof(double));
  double *c = (double*) malloc (width * width * sizeof(double));

  for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      a[i*width+j] = i;
      b[i*width+j] = j;
      c[i*width+j] = 0;
    }
  }

  int size = width*width*sizeof(double);
  double *d_a, *d_b, *d_c;

  cudaMalloc((void **) &d_a, size);
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
cudaMalloc((void **) &d_c, size);

  int blockSize = 2000;
  dim3 dimGrid ((width-1)/blockSize+1, (width-1)/blockSize+1, 1);
  dim3 dimBlock (blockSize, blockSize, 1);

  Tarefa23_cuda<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, width);

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

//    for(int i = 0; i < width; i++) {
//    for(int j = 0; j < width; j++) {
//      printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
//    }
//   }

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
//-----------------PARCODE-----------------//
/*
==7101== NVPROF is profiling process 7101, coTarefa23and: ./Tarefa23
==7101== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==7101== Replaying kernel "Tarefa23$_omp_fn$0" (done)           
==7101== Profiling application: ./Tarefa23
==7101== Profiling result:
==7101== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa23$_omp_fn$0
          1                            warps_launched          72          72          72          72

==7101== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa23$_omp_fn$0
          1                 warp_execution_efficiency                 Warp Execution Efficiency     100.00%     100.00%     100.00%

Tempo do Programa Serial
real  1m50.604s
user  1m15.349s
sys   0m0.108s
Tempo do Programa Paralelo CPU
real  0m40.297s
user  1m11.810s
sys   0m0.315s
Tempo do Programa Paralelo GPU - OpenMP
real  0m1.284s
user  0m0.143s
sys   0m0.814s
Tempo do Programa Paralelo GPU - Cuda
real  0m0.957s
user  0m0.132s
sys   0m0.742s

Speedup Paralelo GPU-OpenMp em relação Paralelo CPU: 2,44
Speedup Paralelo GPU-Cuda em relação Paralelo CPU: 42,107 
*/
//----------------------------------------//


