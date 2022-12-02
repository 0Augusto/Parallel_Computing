/**
 * @author Henrique Augusto Rodrigues
 * Matricula: XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa13.cpp -o Tarefa13
 * Windows:     gcc -o Tarefa13.exe   
 * macbook:     gcc -o Tarefa13.cpp  Tarefa13
 * Para executar em terminal (janela de comandos):   
 * Linux:      ./Tarefa13
 * Windows:      Tarefa13
 * MacBook:    ./Tarefa13
 */

#include <stdio.h>

#define N 42

long fib(long n) {
  long i, j;

  if (n < 2)
    return n;
  else if (n < 30) {
    return fib(n-1) + fib (n-2);
  }
  else {
    #pragma omp parallel sections
    { 
      #pragma omp section 
      i = fib(n-1);
      #pragma omp section 
      j = fib(n-2);
    }
    return i + j;
  }
}

int main() {
  printf("\nFibonacci(%lu) = %lu\n",(long)N,fib((long)N));
}

E possível que o código seja executado, uma vez que, tem a criação de uma section dentro de outra? Considere que, o omp section cria "times" de threads na tentativa de resolver o problema.
Será dividido em equipes de thread e jobs, a criação de sections dentro de outras, é algo possível? Como deve ser feito? 
