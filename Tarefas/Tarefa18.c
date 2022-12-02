/*
 * @authors: Danniel Henrique, Filipe Arthur, Henrique Augusto Rodrigues
 * Matricula: XXXXXX, XXXXXX, XXXXXX
 * Para compilar em terminal (janela de comandos):    
 * Linux:       gcc -O3 -fopt-info-vec-missed Tarefa18.c -o Tarefa18
 * Windows:     gcc -o bubbleParalelo.exe     -fopenmp
 * macbook:     gcc -O3 -fopt-info-vec-missed Tarefa18.c -o Tarefa18
 * Para executar em terminal (janela de comandos):   
 * Linux:      ./Tarefa18   
 * Windows:      Tarefa18
 * MacBook:    ./Tarefa18
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main()
{
    int a[2048], b[2048], c[2048];
    int i;

    // Ter um rand() no numero de iteracoes faz com que os acessos nao sejam sequenciais e em ordem
    for (i = 0; i < rand() % 2048; i++)
    {
        b[i] = 0;
        c[i] = 0;
        
        a[i] = b[i] + c[i];
        
        //i--;    // Tarefa18.c:24:5: note: not vectorized: number of iterations cannot be computed.
               // Essa linha impede o for de ocorrer normalmente se houver apenas as operações sobre os vetores.
        
        // Esse break causa um dos problemas, pois ele quebra a linearidade do loop
        if (i == rand() % 20)
            break; 

        a[i] = b[i] + c[i];

        // Essa atribuicao quebra a vetorização, pois é impossível para o compilador prever exatamente
        // quantas iterações vai ter no loop
        i = rand() % 2048;

        if(i == 3)
            break;

        int aux = rand() % i;

        a[i] = b[i] + c[i];

        // Essas atribuicoes interferem com o fluxo do loop e impedem a previsão do numero de iterações
        i = aux;
        aux = rand() % i;
    }

    return 0;
}
//-----------------MACBOOK-----------------//
/*
*clang: error: unknown argument: '-fopt-info-vec-missed'
*/
//-----------------------------------------//

//-----------------PARCODE-----------------//
/*
*Tarefa18.c:24:21: note: not vectorized: control flow in loop.
*Tarefa18.c:24:21: note: bad loop form.
*Tarefa18.c:18:5: note: not vectorized: not enough data-refs in basic block.
*Tarefa18.c:24:5: note: not vectorized: number of iterations cannot be computed.
*Tarefa18.c:24:5: note: not vectorized: latch block not empty.
*/
//-----------------------------------------//
/* Obs: Tentamos outros códigos, inclusive versões deste, para que apareçam mais erros. 
*Os códigos e erros obtidos se encontram abaixo.
*/

/*
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main()
{
    int a[2048], b[2048], c[2048];
    int i;

    // Ter um rand() no numero de iteracoes faz com que os acessos nao sejam sequenciais e em ordem
    for (i = 0; i < rand() % 2048; i++)
    {
        b[i] = 0;
        c[i] = 0;
    }

    for (i = 0; i < 2048; i++)
    {
        a[i] = b[i] + c[i];

        // Esse break causa um dos problemas, pois ele quebra a linearidade do loop
        if (i == rand() % 20)
            break;
    }

    i = 0;

    while (i < 2048)
    {
        a[i] = b[i] + c[i];

        // Essa atribuicao quebra a vetorização, pois é impossível para o compilador prever exatamente
        // quantas iterações vai ter no loop
        i = rand() % 2048;

        if(i == 3)
            break;
    }

    int aux = rand() % i;

    for(i = 0; i < 2048; i++)
    {
        a[i] = b[i] + c[i];

        // Essas atribuicoes interferem com o fluxo do loop e impedem a previsão do numero de iterações
        i = aux;
        aux = rand() % i;
    }
    
    return 0;
}

Erros:
Tarefa18.c:42:15: missed: not vectorized: multiple exits.
Tarefa18.c:31:11: missed: not vectorized: number of iterations cannot be computed.
Tarefa18.c:20:18: missed: not vectorized: control flow in loop.
Tarefa18.c:10:19: missed: not vectorized: latch block not empty.
*/

//-----------------------------------------//
