/**
 * @author Henrique Augusto Rodrigues
 * Matricula: XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc Tarefa04.c -o Tarefa04 -fopenmp   
 * Windows:     gcc -o Tarefa04.exe   -fopenmp
 * macbook:     gcc -o Tarefa04.c  Tarefa04 -openmp
 * Para executar em terminal (janela de comandos):   
 * Linux:     time ./Tarefa04   
 * Windows:          Tarefa04  
 * MacBook:   time ./Tarefa04
 */
//----------------------- includes --------------------------------------//
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
//-----------------------------------------------------------------------// 


/*
O programa utiliza a diretiva 'omp parallel for collaps(2), de forma que 
o compilador realiza uma 'junção' (collapse) dos dois for e divide as iterações 
entre as threads. Isso é possível uma vez que, a contagem de um for não depende do
outro. A partir do collapse a 'juncao', a divisão atua como um 'omp parallel for' em um 
único for.
Também foi realizado um teste com dois 'omp parallel for' obtendo considerável 
speedup em relação ao cógido serial, mas com menor desempenho que utilizando 
'collapse';
A parelização é bastante eficiente vez que as operaçõe realizadas a partir do 
terceiro for (variável k) são independentes uma das outras em relação aos 'for'
de variáveis 'i' e  'j'. Cada multiplicação e soma dos elementos das linhas 'i'
e colunas 'j' das matrizes são completamente independentes das operações das 
outras linhas e colunas.
*/



void mm(double* a, double* b, double* c, int width) 
{
   //#pragma omp parallel for
   #pragma omp parallel for collapse(2)
   for (int i = 0; i < width; i++) {
      //pragma omp parallel for
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

  mm(a,b,c,width);

    for(int i = 0; i < width; i++) {
    for(int j = 0; j < width; j++) {
      printf("\n c[%d][%d] = %f",i,j,c[i*width+j]);
    }
   }
}
//-----------------MACBOOK-----------------//
/*
*Tempo mm.c Serial
*real   0m30.086s
*user   0m30.067s
*sys    0m0.016s

*Tempo mm.c Paralelo
*real   0m25.885s
*user   0m34.281s
*sys    0m2.223s
*SpeedUp = 30.086/25.885 = 1.16
*/
//-----------------------------------------//

//-----------------PARCODE-----------------//
/*
*Tempo mm.c Serial
*real   2m10.844s
*user   2m10.653s
*sys    0m0.076s

*Tempo mm.c Paralelo
*real   2m27.120s
*user   2m42.393s
*sys    0m17.694s
*SpeedUp = 130.844/157.120 = 0,8328
*/
//-----------------------------------------//
