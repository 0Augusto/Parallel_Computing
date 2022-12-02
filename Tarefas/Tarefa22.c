/*
 * @author(s) Filipe Arthur, Henrique Augusto Rodrigues
 * Matricula: XXXXXX, XXXXXX
 * Tarefa 22
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa22.c -o Tarefa22 -fopenmp   
 * Windows:     gcc -o Tarefa22.exe   -fopenmp
 * macbook:     gcc -o Tarefa22.c  Tarefa22 -openmp
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa22   
 * Windows:          Tarefa22  
 * MacBook:   time ./Tarefa22
 */

/*
//-----------------MACBOOK-----------------//
(Henrique)
*Tempo Tarefa22_sequencial.c
*29,51s user 
*0,15s system 
*99% cpu 
*29,676 total

*Tempo Tarefa22.c

*Tivemos problemas em instalar a biblioteca <omp.h> no MacBook e o programa não pode ser executado

//-----------------------------------------//

//-----------------PARCODE-----------------//
(Filipe)
*Tempo Tarefa22_sequencial.c
*real 1m23.207s (83.207s)
*user 1m19.591s
*sys  0m0.132s

*Tempo Tarefa22.c (Paralelo)

*real 0m5.487s
*user 0m3.515s
*sys  0m1.881s

*SpeedUp = 83.207/5.487 = 15.16

*CUDA Tarefa22.c(Paralelo)
==14627== NVPROF is profiling process 14627, coTarefa22and: ./Tarefa22_par
==14627== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==14627== Replaying kernel "Tarefa22$_omp_fn$0" (done)
==14627== Profiling application: ./Tarefa22_par
==14627== Profiling result:
==14627== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                            warps_launched          72          72          72          72

==14627== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                 warp_execution_efficiency                 Warp Execution Efficiency      86.81%      86.81%      86.81%

//---------------------------------------------------------------------------------------------
(Henrique) 
compilando no parcode com o -O3
//---------------------------------------------------------------------------------------------

//---------------------------------------------------------------------------------------------
*Tempo Tarefa22_sequencial.c
real 2m7.898s (127.898s)
user 2m7.653s
sys  0m0.097s

*Tempo Tarefa22.c (Paralelo)
*real    2m12.659s (132.659s)
*user    1m34.573s
*sys     0m27.676s

*SpeedUp = 127.898/132.659 = 0.96

*CUDA Tarefa22.c(Paralelo)
==13086== NVPROF is profiling process 13086, coTarefa22and: ./Tarefa22
==13086== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==13086== Replaying kernel "Tarefa22$_omp_fn$0" (done)           
==13086== Profiling application: ./Tarefa22
==13086== Profiling result:
==13086== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                            warps_launched          72          72          72          72

==13086== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                 warp_execution_efficiency                 Warp Execution Efficiency     100.00%     100.00%     100.00%

//-----------------------------------------//
*/
//-----bibliotecas-----
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//---------------------

void Tarefa22(double* a, double* b, double* c, int width) 
{
   #pragma omp target map(to: a[0:width*width], b[0:width*width]), map(tofrom: c[0:width*width])
   #pragma omp teams distribute parallel for simd
   for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
         double sum = 0;
         for (int k = 0; k < width; k++) {
	         double x = a[i * width + k];
	         double y = b[k * width + j];
	         sum += x * y;
         }
         c[i * width + j] = sum;
      }
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

  Tarefa22(a,b,c,width);

//    for(int i = 0; i < width; i++) {
//    for(int j = 0; j < width; j++) {
//      printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
//    }
//   }
}
/*
//-----------------MACBOOK-----------------//
(Henrique)
*Tempo Tarefa22_sequencial.c
*29,51s user 
*0,15s system 
*99% cpu 
*29,676 total

*Tempo Tarefa22.c

*Tivemos problemas em instalar a biblioteca <omp.h> no MacBook e o programa não pode ser executado

//-----------------------------------------//

//-----------------PARCODE-----------------//
(Filipe)
*Tempo Tarefa22_sequencial.c


*Tempo Tarefa22.c (Paralelo)

*SpeedUp = 

*CUDA Tarefa22.c(Paralelo)



//---------------------------------------------------------------------------------------------
(Henrique) 
compilando no parcode com o -O3, está dando o seguinte erro
//---------------------------------------------------------------------------------------------
Tarefa22: In function `_fini':
(.fini+0x0): multiple definition of `_fini'
/usr/lib/x86_64-linux-gnu/crti.o:(.fini+0x0): first defined here
Tarefa22: In function `data_start':
(.data+0x0): multiple definition of `__data_start'
/usr/lib/x86_64-linux-gnu/crt1.o:(.data+0x0): first defined here
Tarefa22: In function `data_start':
(.data+0x8): multiple definition of `__dso_handle'
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtbegin.o:(.data+0x0): first defined here
Tarefa22:(.rodata+0x0): multiple definition of `_IO_stdin_used'
/usr/lib/x86_64-linux-gnu/crt1.o:(.rodata.cst4+0x0): first defined here
Tarefa22: In function `_start':
(.text+0x0): multiple definition of `_start'
/usr/lib/x86_64-linux-gnu/crt1.o:(.text+0x0): first defined here
Tarefa22: In function `main':
(.text+0x207): multiple definition of `main'
/tmp/ccI3jj52.o:Tarefa22.c:(.text.startup+0x0): first defined here
Tarefa22: In function `_edata':
(.gnu.offload_funcs+0x8): multiple definition of `__offload_var_table'
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtoffloadbegin.o:(.gnu.offload_vars+0x0): first defined here
Tarefa22:(.gnu.offload_funcs+0x0): multiple definition of `__offload_func_table'
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtoffloadbegin.o:(.gnu.offload_funcs+0x0): first defined here
Tarefa22: In function `_init':
(.init+0x0): multiple definition of `_init'
/usr/lib/x86_64-linux-gnu/crti.o:(.init+0x0): first defined here
Tarefa22: In function `Tarefa22':
(.text+0xd2): multiple definition of `Tarefa22'
/tmp/ccI3jj52.o:Tarefa22.c:(.text+0x1a0): first defined here
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtend.o:(.tm_clone_table+0x0): multiple definition of `__TMC_END__'
Tarefa22:(.gnu.offload_funcs+0x0): first defined here
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtoffloadend.o:(.gnu.offload_funcs+0x0): multiple definition of `__offload_funcs_end'
Tarefa22:(.gnu.offload_funcs+0x8): first defined here
/usr/local/install/bin/../lib/gcc/x86_64-pc-linux-gnu/8.0.0/crtoffloadend.o:(.gnu.offload_vars+0x0): multiple definition of `__offload_vars_end'
Tarefa22:(.gnu.offload_funcs+0x8): first defined here
/tmp/cczpTzR2.crtoffloadtable.o:(.rodata+0x0): multiple definition of `__OFFLOAD_TABLE__'
Tarefa22:(.rodata+0x20): first defined here
/usr/bin/ld: error in Tarefa22(.eh_frame); no .eh_frame_hdr table will be created.
collect2: error: ld returned 1 exit status
//---------------------------------------------------------------------------------------------
*Tempo Tarefa22_sequencial.c
real 2m7.898s
user 2m7.653s
sys  0m0.097s

*Tempo Tarefa22.c (Paralelo)
*real    2m12.659s
*user    1m34.573s
*sys     0m27.676s

*SpeedUp = 2 7.898/2 12.659 = 

*CUDA Tarefa22.c(Paralelo)
==13086== NVPROF is profiling process 13086, coTarefa22and: ./Tarefa22
==13086== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==13086== Replaying kernel "Tarefa22$_omp_fn$0" (done)           
==13086== Profiling application: ./Tarefa22
==13086== Profiling result:
==13086== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                            warps_launched          72          72          72          72

==13086== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: Tarefa22$_omp_fn$0
          1                 warp_execution_efficiency                 Warp Execution Efficiency     100.00%     100.00%     100.00%

//-----------------------------------------//
*/