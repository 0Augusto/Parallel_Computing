/*
Neste código, usamos a função dispatch_apply para executar a iteração do loop em paralelo. Cada iteração é executada em uma thread diferente na fila global de despacho (queue), permitindo a paralelização. A função dispatch_apply recebe o número total de iterações (no caso, 2) e um bloco de código para executar em paralelo.
*/

#include <stdio.h>
#include <dispatch/dispatch.h>

int main() {
    // Obtendo uma fila global de despacho com prioridade padrão
    dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
    
    // Usando dispatch_apply para executar um bloco de código em paralelo 2 vezes
    dispatch_apply(2, queue, ^(size_t tid) {
        // Início do loop for que vai de 1 a 3
        for (int i = 1; i <= 3; i++) {
            // Imprimindo o identificador da thread (tid) e o valor de i
            printf("[PRINT1] T%zu = %d\n", tid, i);
            printf("[PRINT2] T%zu = %d\n", tid, i);
        }
        // Fim do loop for
    }); // Fim do dispatch_apply
    
    // Retornando 0 para indicar que a execução foi concluída sem erros
    return 0;
} // Fim da função main

