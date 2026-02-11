/**
 * charm_calculator_m1.c
 * Cálculo otimizado de eficácia de charms do Tibia para Apple Silicon M1
 * Com suporte para armas, munições, criaturas e runas personalizadas
 * Compilar: clang -O3 -framework Accelerate charm_calculator_m1.c -o charm_calculator
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <unistd.h>
#include <mach/mach_time.h>

// ==================== CONSTANTES E DEFINIÇÕES ====================
#define MAX_NOME 64
#define MAX_CRIATURAS 20
#define MAX_RUNAS 10
#define MAX_ARMAS 10
#define MAX_MUNICOES 10

// Tipos de runas (imbuements)
typedef enum {
    RUNA_CRITICO,
    RUNA_PODER,
    RUNA_VAMPIRISMO,
    RUNA_CAÇA,
    RUNA_PRECISÃO
} TipoRuna;

// Níveis de runa
typedef enum {
    NIVEL_BASICO = 1,
    NIVEL_INTRINCADO = 2,
    NIVEL_PODEROSO = 3
} NivelRuna;

// ==================== ESTRUTURAS DE DADOS ====================
typedef struct {
    char nome[MAX_NOME];
    int hp;
    float defesa_fisica;
    float fraqueza_elemental;
    int quantidade; // Quantos desse bicho na hunt
} Criatura;

typedef struct {
    char nome[MAX_NOME];
    int ataque_base;
    float multiplicador_elemental;
} Arma;

typedef struct {
    char nome[MAX_NOME];
    int ataque_extra;
    float multiplicador_elemental;
} Municao;

typedef struct {
    TipoRuna tipo;
    NivelRuna nivel;
    float porcentagem_efeito; // Varia conforme nível
    int ativo; // 1 = ativa, 0 = inativa
} Runa;

typedef struct {
    Criatura criatura;
    Arma arma;
    Municao municao;
    Runa runas[MAX_RUNAS];
    int num_runas_ativas;
    int nivel_jogador;
    int skill_distancia;
    int ataque_equipamento;
    float resultado_lowblow;
    float resultado_elemental;
    float dm_jogador;
    char charm_recomendado[32];
} CalculoResultado;

// ==================== BANCO DE DADOS ====================
// Tabela de bônus por tipo e nível de runa
static const float BONUS_RUNA[5][3] = {
    // Crítico: chance extra de crítico (acumula com Low Blow)
    {0.05f, 0.10f, 0.15f},  // Básico, Intrincado, Poderoso
    
    // Poder: aumento de dano físico
    {0.02f, 0.04f, 0.06f},  // +2%, +4%, +6%
    
    // Vampirismo: não afeta dano direto, apenas vida
    {0.0f, 0.0f, 0.0f},
    
    // Caça: dano extra a monstros
    {0.03f, 0.05f, 0.07f},  // +3%, +5%, +7%
    
    // Precisão: aumento de chance de acerto (não afeta dano máximo)
    {0.02f, 0.04f, 0.06f}
};

// ==================== FUNÇÕES DE ENTRADA DO USUÁRIO ====================
void limpar_buffer() {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

void listar_opcoes(const char* titulo, const char* opcoes[], int num_opcoes) {
    printf("\n%s\n", titulo);
    for (int i = 0; i < num_opcoes; i++) {
        printf("%d. %s\n", i + 1, opcoes[i]);
    }
}

void configurar_arma(Arma *arma) {
    printf("\n=== CONFIGURAR ARMA ===\n");
    
    const char* armas_disponiveis[] = {
        "Crystalline Crossbow (55 atk)",
        "Warsinger Bow (52 atk)",
        "Bow of Cataclysm (58 atk)",
        "Crossbow of Destruction (50 atk)",
        "Other (custom)"
    };
    
    listar_opcoes("Selecione a arma:", armas_disponiveis, 5);
    
    int escolha;
    scanf("%d", &escolha);
    limpar_buffer();
    
    switch(escolha) {
        case 1:
            strcpy(arma->nome, "Crystalline Crossbow");
            arma->ataque_base = 55;
            break;
        case 2:
            strcpy(arma->nome, "Warsinger Bow");
            arma->ataque_base = 52;
            break;
        case 3:
            strcpy(arma->nome, "Bow of Cataclysm");
            arma->ataque_base = 58;
            break;
        case 4:
            strcpy(arma->nome, "Crossbow of Destruction");
            arma->ataque_base = 50;
            break;
        case 5:
            printf("Nome da arma: ");
            fgets(arma->nome, MAX_NOME, stdin);
            arma->nome[strcspn(arma->nome, "\n")] = 0;
            printf("Ataque base: ");
            scanf("%d", &arma->ataque_base);
            limpar_buffer();
            break;
        default:
            strcpy(arma->nome, "Crystalline Crossbow");
            arma->ataque_base = 55;
    }
    
    arma->multiplicador_elemental = 1.0f;
    printf("Arma selecionada: %s (%d atk)\n", arma->nome, arma->ataque_base);
}

void configurar_municao(Municao *municao) {
    printf("\n=== CONFIGURAR MUNIÇÃO ===\n");
    
    const char* municoes_disponiveis[] = {
        "Diamond Arrow (42 atk)",
        "Bolt (40 atk)",
        "Prismatic Bolt (45 atk)",
        "Other (custom)"
    };
    
    listar_opcoes("Selecione a munição:", municoes_disponiveis, 4);
    
    int escolha;
    scanf("%d", &escolha);
    limpar_buffer();
    
    switch(escolha) {
        case 1:
            strcpy(municao->nome, "Diamond Arrow");
            municao->ataque_extra = 42;
            break;
        case 2:
            strcpy(municao->nome, "Bolt");
            municao->ataque_extra = 40;
            break;
        case 3:
            strcpy(municao->nome, "Prismatic Bolt");
            municao->ataque_extra = 45;
            break;
        case 4:
            printf("Nome da munição: ");
            fgets(municao->nome, MAX_NOME, stdin);
            municao->nome[strcspn(municao->nome, "\n")] = 0;
            printf("Ataque extra: ");
            scanf("%d", &municao->ataque_extra);
            limpar_buffer();
            break;
        default:
            strcpy(municao->nome, "Diamond Arrow");
            municao->ataque_extra = 42;
    }
    
    municao->multiplicador_elemental = 1.0f;
    printf("Munição selecionada: %s (%d atk extra)\n", municao->nome, municao->ataque_extra);
}

void configurar_runas(Runa runas[], int *num_runas_ativas) {
    printf("\n=== CONFIGURAR RUNAS (IMBUEMENTS) ===\n");
    
    const char* tipos_runa[] = {
        "Crítico (aumenta chance de crítico)",
        "Poder (aumenta dano físico)",
        "Vampirismo (rouba vida)",
        "Caça (dano extra a monstros)",
        "Precisão (aumenta chance de acerto)"
    };
    
    *num_runas_ativas = 0;
    
    for (int i = 0; i < MAX_RUNAS; i++) {
        char resposta;
        printf("\nAdicionar runa %d? (s/n): ", i + 1);
        scanf(" %c", &resposta);
        limpar_buffer();
        
        if (resposta != 's' && resposta != 'S') {
            break;
        }
        
        listar_opcoes("Tipo de runa:", tipos_runa, 5);
        
        int tipo;
        scanf("%d", &tipo);
        limpar_buffer();
        
        if (tipo < 1 || tipo > 5) tipo = 1;
        runas[i].tipo = tipo - 1;
        
        printf("\nNível da runa:\n");
        printf("1. Básico\n");
        printf("2. Intrincado\n");
        printf("3. Poderoso\n");
        printf("Escolha: ");
        
        int nivel;
        scanf("%d", &nivel);
        limpar_buffer();
        
        if (nivel < 1 || nivel > 3) nivel = 1;
        runas[i].nivel = nivel;
        
        // Ajusta porcentagem conforme nível
        runas[i].porcentagem_efeito = BONUS_RUNA[runas[i].tipo][nivel - 1];
        runas[i].ativo = 1;
        
        (*num_runas_ativas)++;
        
        printf("Runa %s (Nível %d) adicionada. Bônus: %.1f%%\n",
               tipos_runa[tipo-1], nivel, runas[i].porcentagem_efeito * 100);
    }
}

void configurar_criaturas(Criatura criaturas[], int *num_criaturas) {
    printf("\n=== CONFIGURAR CRIATURAS DA HUNT ===\n");
    
    const char* criaturas_predefinidas[] = {
        "Fury (1300 HP, 15% def)",
        "High Guard (4300 HP, 35% def)",
        "Falcon Knight (6800 HP, 40% def)",
        "Draken (1900 HP, 20% def, 30% fraco gelo)",
        "Glooth (2900 HP, 25% def)",
        "Custom (personalizar)"
    };
    
    *num_criaturas = 0;
    
    for (int i = 0; i < MAX_CRIATURAS; i++) {
        printf("\n--- Criatura %d ---\n", i + 1);
        listar_opcoes("Selecione a criatura:", criaturas_predefinidas, 6);
        
        int escolha;
        scanf("%d", &escolha);
        limpar_buffer();
        
        if (escolha == 6) {
            printf("Nome da criatura: ");
            fgets(criaturas[i].nome, MAX_NOME, stdin);
            criaturas[i].nome[strcspn(criaturas[i].nome, "\n")] = 0;
            
            printf("HP: ");
            scanf("%d", &criaturas[i].hp);
            
            printf("Defesa física (0.0-1.0): ");
            scanf("%f", &criaturas[i].defesa_fisica);
            
            printf("Fraqueza elemental (1.0 = neutro): ");
            scanf("%f", &criaturas[i].fraqueza_elemental);
        } else {
            switch(escolha) {
                case 1:
                    strcpy(criaturas[i].nome, "Fury");
                    criaturas[i].hp = 1300;
                    criaturas[i].defesa_fisica = 0.15f;
                    criaturas[i].fraqueza_elemental = 1.0f;
                    break;
                case 2:
                    strcpy(criaturas[i].nome, "High Guard");
                    criaturas[i].hp = 4300;
                    criaturas[i].defesa_fisica = 0.35f;
                    criaturas[i].fraqueza_elemental = 1.0f;
                    break;
                case 3:
                    strcpy(criaturas[i].nome, "Falcon Knight");
                    criaturas[i].hp = 6800;
                    criaturas[i].defesa_fisica = 0.40f;
                    criaturas[i].fraqueza_elemental = 1.0f;
                    break;
                case 4:
                    strcpy(criaturas[i].nome, "Draken");
                    criaturas[i].hp = 1900;
                    criaturas[i].defesa_fisica = 0.20f;
                    criaturas[i].fraqueza_elemental = 1.3f;
                    break;
                case 5:
                    strcpy(criaturas[i].nome, "Glooth");
                    criaturas[i].hp = 2900;
                    criaturas[i].defesa_fisica = 0.25f;
                    criaturas[i].fraqueza_elemental = 1.1f;
                    break;
                default:
                    strcpy(criaturas[i].nome, "Fury");
                    criaturas[i].hp = 1300;
                    criaturas[i].defesa_fisica = 0.15f;
                    criaturas[i].fraqueza_elemental = 1.0f;
            }
        }
        
        printf("Quantidade na hunt: ");
        scanf("%d", &criaturas[i].quantidade);
        limpar_buffer();
        
        (*num_criaturas)++;
        
        char continuar;
        printf("Adicionar outra criatura? (s/n): ");
        scanf(" %c", &continuar);
        limpar_buffer();
        
        if (continuar != 's' && continuar != 'S') {
            break;
        }
    }
}

// ==================== CÁLCULOS MATEMÁTICOS OTIMIZADOS ====================
/**
 * Calcula dano médio considerando runas (imbuements)
 */
static inline float calcular_dano_medio_com_runas(int nivel, int skill, int ataque_total,
                                                  Runa runas[], int num_runas_ativas) {
    const float FATOR_PALADINO = 1.1f;
    
    // Base: (nivel/6 + skill + ataque) * fator_vocacao
    float inputs[3] = {nivel / 6.0f, (float)skill, (float)ataque_total};
    float produto = 0.0f;
    
    vDSP_dotpr(inputs, 1, (float[]){1.0f, 1.0f, 1.0f}, 1, &produto, 3);
    
    float dano_base = produto * FATOR_PALADINO;
    
    // Aplica bônus das runas
    float multiplicador_runas = 1.0f;
    float chance_critico_extra = 0.0f;
    
    for (int i = 0; i < num_runas_ativas; i++) {
        if (!runas[i].ativo) continue;
        
        switch(runas[i].tipo) {
            case RUNA_CRITICO:
                chance_critico_extra += runas[i].porcentagem_efeito;
                break;
            case RUNA_PODER:
            case RUNA_CAÇA:
                multiplicador_runas += runas[i].porcentagem_efeito;
                break;
            case RUNA_PRECISÃO:
                // Precisão não afeta dano máximo, apenas chance de acerto
                // Para simplificar, consideramos pequeno aumento de dano efetivo
                multiplicador_runas += runas[i].porcentagem_efeito * 0.5f;
                break;
            default:
                break;
        }
    }
    
    // Crítico base (10%) + runa crítico
    float chance_critico_base = 0.10f + chance_critico_extra;
    float multiplicador_critico = 1.0f + (0.5f * chance_critico_base);
    
    return dano_base * multiplicador_runas * multiplicador_critico;
}

/**
 * Calcula eficácia do Low Blow considerando runa de crítico
 */
static inline float calcular_lowblow_com_runas(float dano_medio, Runa runas[], int num_runas_ativas) {
    float chance_critico_extra = 0.0f;
    
    for (int i = 0; i < num_runas_ativas; i++) {
        if (runas[i].ativo && runas[i].tipo == RUNA_CRITICO) {
            chance_critico_extra += runas[i].porcentagem_efeito;
        }
    }
    
    // Low Blow adiciona 10% de chance crítica
    float chance_critico_total = 0.10f + chance_critico_extra + 0.10f;
    float multiplicador = 1.0f + (0.5f * chance_critico_total);
    
    return dano_medio * multiplicador;
}

/**
 * Calcula eficácia de charm elemental
 */
static inline float calcular_elemental_com_runas(float dano_medio, int hp_criatura,
                                                 float fraqueza, Runa runas[], int num_runas_ativas) {
    float bonus_caca = 0.0f;
    
    for (int i = 0; i < num_runas_ativas; i++) {
        if (runas[i].ativo && runas[i].tipo == RUNA_CAÇA) {
            bonus_caca += runas[i].porcentagem_efeito;
        }
    }
    
    return (dano_medio * fraqueza * (1.0f + bonus_caca)) + (hp_criatura * 0.005f);
}

// ==================== PROCESSAMENTO PARALELO ====================
void processar_lote_paralelo(CalculoResultado* resultados, size_t total,
                             int nivel, int skill, int ataque_eq) {
    
    int nucleos = (int)sysconf(_SC_NPROCESSORS_ONLN);
    size_t lote_size = total / nucleos + 1;
    
    // CORREÇÃO: Usar dispatch_group em vez de semáforo e operações atômicas
    dispatch_group_t group = dispatch_group_create();
    
    for (int nucleo = 0; nucleo < nucleos; nucleo++) {
        size_t inicio = nucleo * lote_size;
        size_t fim = (inicio + lote_size) < total ? (inicio + lote_size) : total;
        
        if (inicio >= total) break;
        
        dispatch_group_async(group, dispatch_get_global_queue(
            (nucleo < 4) ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, 0), ^{
            
            for (size_t i = inicio; i < fim; i++) {
                CalculoResultado* r = &resultados[i];
                
                int ataque_total = r->arma.ataque_base + r->municao.ataque_extra + ataque_eq;
                
                r->dm_jogador = calcular_dano_medio_com_runas(
                    nivel, skill, ataque_total,
                    r->runas, r->num_runas_ativas);
                
                r->dm_jogador *= (1.0f - r->criatura.defesa_fisica);
                
                float lb = calcular_lowblow_com_runas(r->dm_jogador, r->runas, r->num_runas_ativas);
                float el = calcular_elemental_com_runas(r->dm_jogador,
                    r->criatura.hp, r->criatura.fraqueza_elemental,
                    r->runas, r->num_runas_ativas);
                
                r->resultado_lowblow = lb;
                r->resultado_elemental = el;
                
                // Regra da comunidade para decidir se Low Blow é melhor
                int lowblow_melhor = (r->criatura.hp * 0.005f) < (r->dm_jogador * 0.04f);
                
                // Escolhe o charm com maior dano médio, considerando a regra
                if (lowblow_melhor && (lb > el)) {
                    snprintf(r->charm_recomendado, 32, "LOW BLOW");
                } else {
                    snprintf(r->charm_recomendado, 32, "ELEMENTAL");
                }
            }
        });
    }
    
    // Aguarda a conclusão de todos os blocos
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);
}

// ==================== FUNÇÃO PRINCIPAL ====================
int main() {
    printf("=== CALCULADOR LOW BLOW TIBIA - Apple M1 ===\n");
    printf("Configuração Personalizada\n");
    
    // Configurações do jogador
    int nivel, skill_dist, ataque_equip;
    
    printf("\n=== CONFIGURAÇÃO DO JOGADOR ===\n");
    printf("Nível do personagem: ");
    scanf("%d", &nivel);
    printf("Skill de Distance Fighting: ");
    scanf("%d", &skill_dist);
    printf("Ataque extra de equipamentos (helmet, armor, etc.): ");
    scanf("%d", &ataque_equip);
    limpar_buffer();
    
    // Configurar arma
    Arma arma_usuario;
    configurar_arma(&arma_usuario);
    
    // Configurar munição
    Municao municao_usuario;
    configurar_municao(&municao_usuario);
    
    // Configurar runas
    Runa runas_usuario[MAX_RUNAS];
    int num_runas_ativas = 0;
    configurar_runas(runas_usuario, &num_runas_ativas);
    
    // Configurar criaturas
    Criatura criaturas_hunt[MAX_CRIATURAS];
    int num_criaturas = 0;
    configurar_criaturas(criaturas_hunt, &num_criaturas);
    
    if (num_criaturas == 0) {
        printf("Nenhuma criatura configurada. Encerrando.\n");
        return 0;
    }
    
    // Preparar resultados
    size_t total_combinacoes = num_criaturas;
    CalculoResultado* resultados = malloc(total_combinacoes * sizeof(CalculoResultado));
    
    if (!resultados) {
        fprintf(stderr, "Erro: Falha ao alocar memória.\n");
        return 1;
    }
    
    // Preencher combinações
    for (size_t i = 0; i < total_combinacoes; i++) {
        resultados[i].criatura = criaturas_hunt[i];
        resultados[i].arma = arma_usuario;
        resultados[i].municao = municao_usuario;
        resultados[i].nivel_jogador = nivel;
        resultados[i].skill_distancia = skill_dist;
        resultados[i].ataque_equipamento = ataque_equip;
        resultados[i].num_runas_ativas = num_runas_ativas;
        
        // Copiar runas
        for (int j = 0; j < num_runas_ativas; j++) {
            resultados[i].runas[j] = runas_usuario[j];
        }
    }
    
    // Processar em paralelo
    uint64_t inicio = mach_absolute_time();
    
    processar_lote_paralelo(resultados, total_combinacoes,
                           nivel, skill_dist, ataque_equip);
    
    uint64_t fim = mach_absolute_time();
    
    mach_timebase_info_data_t timebase;
    mach_timebase_info(&timebase);
    double tempo_ns = (double)(fim - inicio) * timebase.numer / timebase.denom;
    
    // Exibir resultados
    printf("\n=== RESULTADOS DA HUNT ===\n");
    printf("Jogador: Nível %d, Skill %d\n", nivel, skill_dist);
    printf("Arma: %s, Munição: %s\n", arma_usuario.nome, municao_usuario.nome);
    printf("Runas ativas: %d\n", num_runas_ativas);
    printf("Tempo processamento: %.2f μs\n", tempo_ns / 1000.0);
    printf("\n");
    
    int lowblow_recomendados = 0;
    int elemental_recomendados = 0;
    
    for (size_t i = 0; i < total_combinacoes; i++) {
        printf("Criatura: %s (x%d)\n",
               resultados[i].criatura.nome,
               resultados[i].criatura.quantidade);
        printf("  Dano Médio: %.1f\n", resultados[i].dm_jogador);
        printf("  Low Blow: %.1f | Elemental: %.1f\n",
               resultados[i].resultado_lowblow,
               resultados[i].resultado_elemental);
        printf("  RECOMENDADO: %s\n\n", resultados[i].charm_recomendado);
        
        if (strcmp(resultados[i].charm_recomendado, "LOW BLOW") == 0) {
            lowblow_recomendados++;
        } else {
            elemental_recomendados++;
        }
    }
    
    printf("=== RESUMO GERAL ===\n");
    printf("Low Blow recomendado para: %d criaturas\n", lowblow_recomendados);
    printf("Elemental recomendado para: %d criaturas\n", elemental_recomendados);
    
    if (lowblow_recomendados > elemental_recomendados) {
        printf("\nRECOMENDAÇÃO FINAL: LOW BLOW é mais vantajoso nesta hunt!\n");
    } else if (elemental_recomendados > lowblow_recomendados) {
        printf("\nRECOMENDAÇÃO FINAL: Charm ELEMENTAL é mais vantajoso!\n");
    } else {
        printf("\nRECOMENDAÇÃO FINAL: Ambos são equivalentes nesta hunt.\n");
    }
    
    free(resultados);
    
    printf("\n=== Fim dos cálculos ===\n");
    return 0;
}
