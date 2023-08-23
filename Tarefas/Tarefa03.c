/*
 * Adapted from: http://w...content-available-to-author-only...s.org/sieve-of-eratosthenes
 * @Grupo: Filipe Arthur, Henrique Augusto,Luam Gonçalves 
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa03.c -o Tafera03 -fopenmp -lm   * Windows:     gcc -o Tarefa03.exe   -fopenmp -lm
 * macbook:     gcc -o Tarefa03.c -o Tarefa03 -openmp ou gcc -openmp Tarefa03.c -o Tarefa03
 * Para executar em terminal (janela de comandos):   
 * Linux:     ./Tarefa03   
 * Windows:     Tarefa03  
 * MacBook:   ./Tarefa03
*/
//----------------------- includes --------------------------------------//
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>
 //-----------------------------------------------------------------------// 
int sieveOfEratosthenes(int n)
{
   // Create a boolean array "prime[0..n]" and initialize
   // all entries it as true. A value in prime[i] will
   // finally be false if i is Not a prime, else true.
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);

   memset(prime, true,(n+1)*sizeof(bool));

   #pragma omp parallel for schedule(dynamic) num_threads(4)
   for (int p = 2; p <= sqrt_n; p++)
   {
       // If prime[p] is not changed, then it is a prime
       if (prime[p] == true)
       {
           // Update all multiples of p
         
         #pragma omp parallel for num_threads(4) 
         for (int i=p*2; i<=n; i += p)
           prime[i] = false;
        }
    }

    // count prime numbers
   #pragma omp parallel for reduction (+:primes) num_threads(4)
   for (int p=2; p<=n; p++)
       if (prime[p])
         primes++;

    return(primes);
}

int main()
{
  // para armazenar o tempo de execução do código
    double time_spent = 0.0;
 
    clock_t begin = clock();
  
    int n = 100000000;
    printf("%d\n",sieveOfEratosthenes(n));

  // calcula o tempo decorrido encontrando a diferença (end - begin) e dividindo a diferença por CLOCKS_PER_SEC para converter em segundos
    
    clock_t end = clock();
    time_spent += (double)(end - begin) / CLOCKS_PER_SEC;
 
    printf("The elapsed time is %f seconds\n", time_spent);
    return 0;
} 

/**
 * @version 1.0 - Código fornecido pelo professor e com saida padrão:
 * Nao ha tempo de execucao informado no codigo original      
 * print: 5761455  
 * Tempo Repl.it 1.187835 second
 * Tempo servidor(parcode) 1.153447 seconds      
 *-----------------------------------------------------------------
 * @version 2.0 - Calculando o tempo real de execucao do programa
 * '#O tempo resultante foi:
 * print: 5761455 
 * tempo Repl.it 1.089521 seconds     
 * Tempo servidor(parcode) 8.944272 seconds            
 */

/* Comparação entre os tempos de execução:
 * No Repl.it, plataforma onde realizamos a tarefa, os códigos
 * sequencial e paralelo apresentaram diferenças muito pequenas
 * no tempo de execução. Enquanto isso, no servidor a diferença
 * maior, com o código paralelo apresentando uma demora maior na 
 * execução. Não descobrimos a causa dessa diferença.
 */
