/**
 * @author Danniel Henrique, Filipe Arthur, Henrique Augusto Rodrigues
 * Matricula: XXXXXX, XXXXXX, XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa09.c -o Tarefa09 -fopenmp   
 * Windows:     gcc -o Tarefa09.exe     -fopenmp
 * macbook:     gcc -o sieve.c  Tarefa09 -openmp 
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa09   
 * Windows:          Tarefa09  
 * MacBook:   time ./Tarefa09
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main() 
{
   omp_set_num_threads(2);   

   int i, j, n = 30000; 
 
   // Allocate input, output and position arrays
   int *in = (int*) calloc(n, sizeof(int));
   int *pos = (int*) calloc(n, sizeof(int));   
   int *out = (int*) calloc(n, sizeof(int));   
 
   #pragma omp parallel for 
   for(i=0; i < n; i++)
      in[i] = n-i;  
 
   // Print input array
   //   for(i=0; i < n; i++) 
   //      printf("%d ",in[i]);
 
   #pragma omp parallel for collapse(2) schedule(guided)
   for(i=0; i < n; i++) 
      for(j=0; j < n; j++)
	     if(in[i] > in[j]) 
            pos[i]++;	
 
   #pragma omp parallel for 
   for(i=0; i < n; i++) 
      out[pos[i]] = in[i];
 
   // print output array
   //   for(i=0; i < n; i++) 
   //      printf("%d ",out[i]);
 
   #pragma omp parallel for  
   for(i=0; i < n; i++)
      if(i+1 != out[i]) 
      {
         printf("test failed\n");
         exit(0);
      }
 
   printf("test passed\n"); 
} 
//-----------------MACBOOK-----------------//
/*
*Tempo Tarefa09.c Serial
1,54s user 
0,01s system 
74% cpu 
2,066 total

*Tempo Tarefa09.c Paralelo
*1,54s user 
*0,01s system 
*71% cpu 
*2,154 total

SpeedUP: 2,066/2,154 = 0,95
*/
//-----------------------------------------//

//-----------------PARCODE-----------------//
/*
*Tempo Tarefa09.c Serial
*real	0m4.499s
*user	0m4.490s
*sys	0m0.004s


*Tempo Tarefa09.c Paralelo
*real	0m4.374s
*user	0m4.367s
*sys	0m0.004s

*SpeedUP: 1,028

*Tempo Tarefa09.c collapse(2) schedule(guided)
*real	0m2.684s
*user	0m4.970s
*sys	0m0.009s
*Tempo Danniel
*SEQUENCIAL
*real    0m6.538s
*user    0m6.534s
*sys     0m0.000s
*PARALELIZADA SEM POLITICA
*real    0m4.394s
*user    0m4.381s
*sys     0m0.004s
*PARALELIZADO COM POLITICA
*real    0m2.644s
*user    0m4.898s
*sys     0m0.008s
*speedUp = 6,53/2,64 = 2,47

*Tempo Filipe
*Sequencial:
*real 5.665s
*user 4.515s
*sys 0s
*Paralelo:
*real 2.655s
*user 4.919s
*sys 0s
*Speed up = 5.665/2.655 = 2.133
*/
//-----------------------------------------//



