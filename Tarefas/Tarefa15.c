/*
* @author Danniel Henrique, Filipe Arthur, Henrique Augusto Rodrigues
* Matricula: XXXXXX, XXXXXX, XXXXXX
* Para compilar em terminal (janela de comandos):
* Linux: gcc Tarefa15.c -o Tarefa15 -fopenmp
* Windows: gcc -o Tarefa15.exe -fopenmp
* macbook: gcc -o sieve.c Tarefa15 -openmp
* Para executar em terminal (janela de comandos):
* Linux: time ./Tarefa15
* Windows: Tarefa15
* MacBook: time ./Tarefa15
*/


#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 10000

void swap(int *a, int *b);

void bubblesort(int arr[], int size);

int main()
{
int a[SIZE];
for(int i = SIZE; i > 0; i--)
{
a[SIZE-i] = i;
}

for(int i = 0; i < SIZE; i++)
printf("%d ", a[i]);

printf("\n\n\n");


bubblesort(a, SIZE);

for(int i = 0; i < SIZE; i++)
printf("%d ", a[i]);

return 0;
}


void swap(int *a, int *b)
{
int aux = *a;
*a = *b;
*b = aux;
}


void bubblesort(int arr[], int size)
{
int i, j, aux;

for(i = 0; i < size-1; i++)
{
aux = i % 2;
#pragma omp parallel for
for(j = aux; j < size-1; j++)
{
if(arr[j] > arr[j+1])
{
swap(&arr[j], &arr[j+1]);
}
}
}
}

//-----------------MACBOOK-----------------//
/*
*Tempo bubbleSequencial.c Serial
*0,00s user
*0,00s system
*46% cpu
*0,003 total

*Tempo Tarefa15.c Paralelo
<omp.h> ainda não consegui resolvê-lo para funcionar no MacBook

SpeedUP: ?
*/
//-----------------------------------------//

//-----------------PARCODE-----------------//
/*
*Tempo Tarefa15.c
*real 0m4.499s
*user 0m4.490s
*sys 0m0.004s


*Tempo bubbleSequencial.c
Alguns erros que, não pude resolver

*SpeedUP: ?
*/
//-----------------------------------------//

/*
//-----------------COMPLEXIDADE-----------------//


Complexidade no modelo RAM: O(N^2)

Complexidade no modelo PRAM: O(N)

//----------------------------------------------//

Observacoes do Professor
Com uma modificação para odd-even, a complexidade do loop interno é 1, uma vez que as comparações são feitas em pares. O loop externo é O(n).
*/
