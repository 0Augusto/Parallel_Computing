/* C implementation Tarefa14 from  http://w...content-available-to-author-only...s.org/quick-sort/*/

/*
 * @author Danniel Henrique, Filipe Arthur, Henrique Augusto Rodrigues
 * Matricula: XXXXXX, XXXXXX, XXXXXX
 * Tarefa 14
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa14.c -o Tarefa14 -fopenmp   
 * Windows:     gcc -o Tarefa14.exe   -fopenmp
 * macbook:     gcc -o Tarefa14.c  Tarefa14 -openmp
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa14   
 * Windows:          Tarefa14  
 * MacBook:   time ./Tarefa14
 */

//----------------------- includes --------------------------------------//
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
//-----------------------------------------------------------------------// 

// A utility function to swap two elements
void swap(int* a, int* b)
{
  int t = *a;
  *a = *b;
  *b = t;
}
 
/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
int partition (int arr[], int low, int high)
{
  int pivot = arr[high];    // pivot
  int i = (low - 1);  // Index of smaller element
 
  for (int j = low; j <= high- 1; j++)
    {
      // If current element is smaller than or
      // equal to pivot
      if (arr[j] <= pivot)
        {
     i++;    // increment index of smaller element
     swap(&arr[i], &arr[j]);
        }
    }
  swap(&arr[i + 1], &arr[high]);
  return (i + 1);
}
 
/* The main function that implements Tarefa14
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */

void Tarefa141(int arr[], int low, int high)
{
  if (low < high)
    {
      /* pi is partitioning index, arr[p] is now
    at right place */
      int pi = partition(arr, low, high);
 
      // Separately sort elements before
      // partition and after partition
      

      Tarefa141(arr, low, pi - 1); 
      Tarefa141(arr, pi + 1, high);

    }
}

void Tarefa14(int arr[], int low, int high)
{
  if (low < high)
    {
      /* pi is partitioning index, arr[p] is now
    at right place */
      int pi = partition(arr, low, high);
 
      // Separately sort elements before
      // partition and after partition
      
      omp_set_num_threads(2);

      #pragma omp parallel sections
      {

      #pragma omp section
      Tarefa141(arr, low, pi - 1);
      #pragma omp section
      Tarefa141(arr, pi + 1, high);

      }
    }
}
 
/* Function to print an array */
void printArray(int arr[], int size)
{
  int i;
  for (i=0; i < size; i++)
    printf("%d ", arr[i]);
  printf("\n");
}
 
// Driver program to test above functions
int main()
{

  int i,n = 10000000;
  int *arr = (int*) malloc(n*sizeof(int));
 
  for(i=0; i < n; i++)
    arr[i] = rand()%n;
 
  Tarefa14(arr, 0, n-1);
//  printf("Sorted array: \n");
//  printArray(arr, n);
  return 0;
}

/*
//-----------------MACBOOK-----------------//
(Henrique)
*Tempo quickSortSequencial.c
*1,47s user 
*0,02s system 
*99% cpu 
*1,492 total

*Tempo Tarefa14.c

*Tivemos problemas em instalar a biblioteca <omp.h> no MacBook e o programa não pode ser executado

//-----------------------------------------//

//-----------------PARCODE-----------------//
(Danniel)
*Tempo quickSortSequencial.c
*real 0m4.433s
*user 0m3.898s
*sys  0m0.020s

*Tempo Tarefa14.c
*real 0m2.769s
*user 0m3.912s
*sys  0m0.024s
*SpeedUp = 4.433/2.769 = 1.6

(Filipe)
*Tempo quickSortSequencial.c
*real 0m4.280s                               
*user 0m4.240s                                
*sys  0m0.028s

*Tempo Tarefa14.c
*real    0m2.791s
*user    0m3.956s
*sys     0m0.012s
*SpeedUp = 4.280/2.791 = 1.54

(Henrique)
*Tempo quickSortSequencial.c
*real	0m5.869s
*user	0m5.838s
*sys	0m0.024s

*Tempo Tarefa14.c
*real	0m2.938s
*user	0m4.201s
*sys	0m0.028s
*SpeedUp = 5.869/2.938 = 1.99
//-----------------------------------------//

*/