/**
 * @author Henrique Augusto Rodrigues
 * Matricula: XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa07.c -o Tarefa07 -fopenmp -lm   
 * Windows:     gcc -o Tarefa07.exe     -fopenmp -lm
 * macbook:     gcc -o sieve.c  Tarefa07 -openmp -lm
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa07   
 * Windows:          Tarefa07  
 * MacBook:   time ./Tarefa07
 */

/*
No 'for' externo da linha 47 e interno de linha 61 não têm variáveis
compartilhadas que possam, em razão da paralelização, fazer uso da diretiva
'reduction'. Nesses dois 'for', como o trabalho realizado em cada iteração
é independente, o melhor é simplesmente paralizar as interações.
No entanto, como cada iteração do 'for' externo tem uma carga de trabalho
diferente, utiliza-se a diretiva 'schedule(dynamic)' para que haja uma 
distribuição da carga de trabalho de forma dinâmica. Assim, uma thread com
menos trabalho não fica ociosa quando termina seu 'chunk', ela busca mais
trabalho.
Por outro lado, o 'for' de linha 61 acumula o resultado na variável 'primes'.
Assim, a diretiva 'reduction(+:primes)' deve ser utilizada, para que o resultado
de cada thread seja ao final acumulado na variável.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

int sieveOfEratosthenes(int n)
{
   // Create a boolean array "prime[0..n]" and initialize
   // all entries it as true. A value in prime[i] will
   // finally be false if i is Not a prime, else true.
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);

   memset(prime, true,(n+1)*sizeof(bool));

   #pragma omp parallel for schedule(dynamic) 
   for (int p = 2; p <= sqrt_n; p++)
   {
       // If prime[p] is not changed, then it is a prime
       if (prime[p] == true)
       {
           // Update all multiples of p
            #pragma omp parallel for
           for (int i= p * 2; i <= n; i += p)
           prime[i] = false;
        }
    }

    // count prime numbers
   #pragma omp parallel for reduction(+:primes)
    for (int p = 2; p <=n; p++)
       if (prime[p])
         primes++;

    return(primes);
}

int main()
{
   int n = 100000000;
   printf("%d\n",sieveOfEratosthenes(n));
   return 0;
} 
//-----------------MACBOOK-----------------//
/*
*Tempo sieve.c Serial
*1,02s user 
*0,02s system 
*66% cpu 
*1,571 total

*Tempo sieve.c Paralelo
*1,03s user 
*0,03s system 
*65% cpu 
*1,598 total
*/
//-----------------------------------------//

//-----------------PARCODE-----------------//
/*
*Tempo sieve.c Serial
*real	0m4.073s
*user	0m3.992s
*sys	0m0.072s

*Tempo sieve.c Paralelo
*real	0m2.310s
*user	0m8.730s
*sys	0m0.096s
*/
//-----------------------------------------//
