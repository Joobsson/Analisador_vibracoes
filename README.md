"""
UFSJ-CAP — Sistema Integrado de Diagnóstico e Prognóstico de Rolamentos

Fluxo completo do sistema:

1. Programa principal (este arquivo):
   - Carrega imagem de espectro ou arquivo CSV
   - Digitaliza o gráfico → converte pixels em valores reais de frequência × amplitude
   - Calcula frequências características de defeito (BPFO, BPFI, BSF, FTF)
   - Executa diagnóstico básico + modelo de machine learning treinado
   - Gera visualização com harmônicos marcados
   - Exporta dados discretizados + metadados para Excel (.xlsx)

2. Programa de análise avançada e prognóstico (segundo script):
   - Lê o arquivo .xlsx gerado pelo programa principal
   - Realiza rastreamento automático de harmônicos em múltiplas ordens
   - Calcula métricas globais (RMS espectral, fator de crista, energia por bandas)
   - Aplica regras de severidade e prognósticos qualitativos por tipo de defeito
   - Gera laudo preditivo completo com projeção de evolução de falha

Objetivo:
Auxiliar engenheiros de manutenção e confiabilidade na identificação precoce
de defeitos em rolamentos de esferas (modelos 6203ZZ e 6204ZZ) a partir de
espectros de vibração digitalizados ou medidos.

Uso recomendado:
Execute primeiro o programa principal → exporte o Excel → abra no analisador
para obter o laudo técnico conclusivo com prognóstico.

Data da última atualização do sistema: Março/2026
"""
