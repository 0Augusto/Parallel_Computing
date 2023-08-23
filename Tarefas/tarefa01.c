/**
 * @author Henrique Augusto Rodrigues
 * Matricula: XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc tarefa01.c -o tafera01 -fopenmp   
 * Windows:     gcc -o tarefa01.exe   -fopenmp
 * macbook:     clang -o tarefa01 tarefa01.c -fopenmp -lomp
 * Para executar em terminal (janela de comandos):   
 * Linux:     ./tarefa01   
 * Windows:     tarefa01  
 * MacBook:   ./tarefa01
 */

//----------------------- includes --------------------------------------//

#include <stdio.h>
#include <omp.h>

//-----------------------------------------------------------------------// 

int main()
{
    int i;
    #pragma omp parallel num_threads(2)// seta o numero de threads em 2 
    {
        int tid = omp_get_thread_num(); // lê o identificador da thread 
	#pragma omp for	
	for(i = 1; i <= 3; i++) 
        {
           printf("[PRINT1] T%d = %d \n",tid,i);
           printf("[PRINT2] T%d = %d \n",tid,i);
        }
    }
}

/**
 * @version 1.0 - Código fornecido pelo professor e com saida padrão:
 *       [PRINT1] T0 = 1
 *       [PRINT2] T0 = 1
 *       [PRINT1] T0 = 2
 *       [PRINT2] T0 = 2
 *       [PRINT1] T0 = 3
 *       [PRINT2] T0 = 3
 *       [PRINT1] T1 = 1
 *       [PRINT2] T1 = 4
 *-------------------------------------------------------------------------	
 * @version 2.0 - Inserindo a linha de código acima do loop 'for' 
 * '#pragma omp for' as saidas resultantes foram:
 *       [PRINT1] T0 = 1
 *       [PRINT2] T0 = 1
 *       [PRINT1] T0 = 2
 *       [PRINT2] T0 = 2
 *       [PRINT1] T1 = 3
 *       [PRINT2] T1 = 3
 */   
