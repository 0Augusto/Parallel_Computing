/*
 * @author(s) Filipe Arthur, Henrique Augusto Rodrigues
 * Matricula: 691337, 675263
 * Tarefa 24
 */
/*
O gasto maior de tempo visto na execução do código em cuda sem memória local compartilhada se dá
em razão do uso do próprio arranjo dos números a ser somado, memória global, para ler e armazenar 
as somas parciais de cada bloco, o que é um pouco mais lento, é um overhead, uma alternativa é, 
um número de blocos maior e reduzir o número de threads menor. No código com memória local, para
cada bloco, não há necessidade, dentro do processamento de cada bloco, de acesso a memória global.
No entanto, o problema de desempenho da solução com memória local é a necessidade de sincronização
das threads. Serão sincronizadas após a cópia dos elementos para a memória local de cada bloco e
novamente depois dentro de cada etapa da soma. No código em cuda sem memória local, somente é 
necessária a sincronização em cada etapa de soma. Por isso a diferença dos códigos em cuda com
e sem memória local não é acentuada.
Pelo tempo gasto pela função "CUDA memcpy HtoD" em comparação com o tempo da função sum_cuda 
é possível verificar que o overhead para transferência dos dados para a GPU é considerável em
comparação com o tempo de computação efetivo. Isso explica porque os tempos serial e multicore
foram melhores. Mas o overhead de cópia da CPU para a GPU e vice-versa, prejudica o desempenho final.
Por fim, o desempenho do código com uso de GPU com openmp teve desempenho similar aos em cuda, 
confirmando que o overhead do uso da GPU prejudica bastante o desempenho do código.
Entre os resultados do serial e do gpu houve uma ligeira mudanca no tempo total e na % da cpu
*/
//-----------------MACBOOK AIR (M1)-----------------//
//sum_serial.c//
Sum = 799999980000000.000000
0,23s user 
0,05s system 
33% cpu 
0,832 total

//sum_gpu.c//
Sum = 799999980000000.000000
0,23s user 
0,05s system 
34% cpu 
0,826 total

//sum_cuda.cu//
Não há a possibilidade do uso do CUDA em arquitetura ARM desde o Rosetta 2,
até onde eu tenha pesquisado.

//sum_multicore.c//
Sum = 799999980000000.000000
0,23s user 
0,05s system 
32% cpu 
0,865 total

SpeedUp = 0.23/0.23 = 1
//-----------------------------------------//

//-----------------PARCODE-----------------//

//Serial//
real    0m0.338s
user    0m0.098s
sys     0m0.237s

//Paralelo sem GPU//
real    0m0.336s
user    0m0.247s
sys     0m0.235s

//GPU//
real    0m2.122s
user    0m0.856s
sys     0m1.174s

SpeedUp: 0.338/0.336 = 1

//NVPROF GPU//
==2812== Profiling application: ./sum_gpu
==2812== Profiling result:
==2812== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: main$_omp_fn$0
          1                            warps_launched          72          72          72          72

==2812== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: main$_omp_fn$0
          1                 warp_execution_efficiency                 Warp Execution Efficiency      99.72%      99.72%      99.72%

//CUDA//
real    0m1.630s
user    0m0.546s
sys     0m1.001s

//NVPROF CUDA//
==3426== Profiling application: ./sum_cuda
==3426== Profiling result:
==3426== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: sum_cuda(double*, double*, int)
          1                            warps_launched     1250016     1250016     1250016     1250016

==3426== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: sum_cuda(double*, double*, int)
          1                 warp_execution_efficiency                 Warp Execution Efficiency      99.15%      99.15%      99.15%

//NVPROF NONSHARED//
==3899== Profiling application: ./sum_noshared
==3899== Profiling result:
==3899== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GT 1030 (0)"
    Kernel: sum_cuda(double*, double*, int)
          1                            warps_launched     1250016     1250016     1250016     1250016

==3899== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GT 1030 (0)"
    Kernel: sum_cuda(double*, double*, int)
          1                 warp_execution_efficiency                 Warp Execution Efficiency      99.82%      99.82%      99.82%
//----------------------------------------------------------------------------------------------------------------------------------//
