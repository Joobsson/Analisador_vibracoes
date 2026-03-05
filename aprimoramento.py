import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import re

def analisar_laudo_tecnico_conclusivo():
    # Inicializa o subsistema de janelas e oculta o frame principal vazio
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "Engenharia de Confiabilidade e Prognóstico", 
        "Selecione o arquivo Excel (.xlsx) contendo a Discretização e os Metadados Físicos."
    )
    
    # 1. Interface de Seleção da Matriz de Dados
    caminho_entrada = filedialog.askopenfilename(
        title="Selecionar Planilha Base",
        filetypes=[("Planilhas Excel", "*.xlsx")]
    )
    
    if not caminho_entrada:
        return # Execução abortada pelo usuário

    try:
        # 2. Leitura vetorial direta do formato Excel
        df_spec = pd.read_excel(caminho_entrada, sheet_name='Discretização Matricial')
        df_meta = pd.read_excel(caminho_entrada, sheet_name='Metadados Físicos')
        
        freqs = df_spec['Frequencia_Hz'].values
        amps = df_spec['Amplitude_g'].values
        
        # Obtenção do limite superior da banda espectral (f_max)
        f_max = np.max(freqs)

        # 3. Extração Dinâmica de Metadados Cinemáticos via Expressões Regulares (Regex)
        def extrair_valor_numerico(parametro):
            valor_str = df_meta.loc[
                df_meta['Parâmetro_Cinemático'].str.contains(parametro, case=False, na=False), 
                'Grandeza_Física'
            ].values[0]
            numeros = re.findall(r"[-+]?\d*\.\d+|\d+", str(valor_str).replace(',', '.'))
            return float(numeros[0]) if numeros else 0.0

        f_1x = extrair_valor_numerico('Frequência Rotacional')
        bpfo = extrair_valor_numerico('BPFO')
        bpfi = extrair_valor_numerico('BPFI')
        bsf = extrair_valor_numerico('BSF')
        ftf = extrair_valor_numerico('FTF')

        # 4. Cálculos Globais de Mecânica Vibratória
        rms_espectral = np.sqrt(np.sum(amps**2))
        pico_maximo = np.max(amps)
        fator_crista = pico_maximo / rms_espectral if rms_espectral > 0 else 0

        # 5. Filtro Lógico por Bandas de Frequência (Spectral Banding)
        b1_mask = (freqs >= 0) & (freqs < f_1x)
        b2_mask = (freqs >= f_1x) & (freqs < 150)
        b3_mask = (freqs >= 150) & (freqs <= f_max)

        energia_bandas = {
            'Banda 1 - Subsíncrona (0 a 1x RPM)': np.sum(amps[b1_mask]),
            'Banda 2 - Síncrona/Componentes (1x RPM a 150 Hz)': np.sum(amps[b2_mask]),
            'Banda 3 - Alta Frequência (> 150 Hz)': np.sum(amps[b3_mask])
        }

        # 6. Sistema Especialista: Avaliação Lógica de Severidade
        def classificar_severidade(amplitude):
            if amplitude < 0.01:
                return "Normal (Sem indício de falha)"
            elif 0.01 <= amplitude < 0.05:
                return "Alerta (Dano incipiente na subsuperfície)"
            elif 0.05 <= amplitude < 0.1:
                return "Crítico (Lascamento/Spalling desenvolvido)"
            else:
                return "Severo (Risco de falha catastrófica iminente)"

        # 7. Sistema de Prognóstico: Projeção de Evolução de Danos na Curva P-F
        def projetar_falha_futura(defeito, amplitude):
            if amplitude < 0.01:
                return "Operação estável. Espera-se apenas desgaste tribológico natural a longo prazo."
            
            elif 0.01 <= amplitude < 0.05:
                if "FTF" in defeito:
                    return "Degradação inicial da gaiola. Risco de aceleração do desgaste por atrito deslizante; possível quebra iminente da estrutura retentora."
                elif "BSF" in defeito:
                    return "Microfissuras no elemento rolante. Evoluirá para travamento intermitente e impacto severo nas pistas externa e interna."
                else:
                    return "Propagação de microfissuras por fadiga (Pitting). Previsão de surgimento de bandas laterais de 1x RPM modulando o espectro a médio prazo."
            
            elif 0.05 <= amplitude < 0.1:
                if "FTF" in defeito or "BSF" in defeito:
                    return "Lascamento avançado (Spalling) nos elementos de translação. Risco altíssimo de travamento cinemático do eixo a curto prazo."
                else:
                    return "Lascamento macroscópico na pista. Espera-se elevação drástica da temperatura, aumento do ruído de base (broadband noise) e aumento de folga radial."
            
            else:
                return "Colapso mecânico generalizado iminente. Risco de soldagem a frio dos rolamentos, perda de concentricidade e destruição do alojamento/eixo."

        # 8. Algoritmo de Rastreamento Dinâmico de Harmônicos (Kinematic Matching Amplo)
        def encontrar_amplitude_pico(f_alvo, tol=0.015):
            mask = (freqs >= f_alvo * (1 - tol)) & (freqs <= f_alvo * (1 + tol))
            if np.any(mask):
                return np.max(amps[mask])
            return 0.0

        analise_harmonica = []
        fcd_dict = {
            '1x RPM (Desbalanceamento)': f_1x, 
            'BPFO (Pista Externa)': bpfo, 
            'BPFI (Pista Interna)': bpfi, 
            'BSF (Corpo Rolante)': bsf, 
            'FTF (Gaiola)': ftf
        }
        
        for defeito, freq_fund in fcd_dict.items():
            if freq_fund <= 0: continue
            
            # Cálculo paramétrico do limite de banda de Nyquist
            h_max = int(f_max // freq_fund)
            
            for h in range(1, h_max + 1):
                f_busca = freq_fund * h
                amp_encontrada = encontrar_amplitude_pico(f_busca)
                
                diagnostico_atual = classificar_severidade(amp_encontrada)
                prognostico_futuro = projetar_falha_futura(defeito, amp_encontrada)
                
                analise_harmonica.append({
                    'Origem Cinemática': defeito,
                    'Ordem Harmônica': f"{h}x",
                    'Frequência Teórica (Hz)': round(f_busca, 2),
                    'Amplitude Extraída (g)': amp_encontrada,
                    'Estado Atual (Severidade)': diagnostico_atual,
                    'Projeção de Degradação Futura': prognostico_futuro
                })

        # 9. Consolidação de Estruturas de Dados (DataFrames)
        df_harmonic = pd.DataFrame(analise_harmonica)
        df_global = pd.DataFrame({
            'Métrica Analítica': ['RMS Espectral Global (g)', 'Pico Máximo (g)', 'Fator de Crista Espectral', 'Banda Espectral (f_max)'],
            'Valor Computado': [rms_espectral, pico_maximo, fator_crista, f_max]
        })
        df_bandas = pd.DataFrame(list(energia_bandas.items()), columns=['Faixa de Espectro', 'Energia Integrada (g)'])

        # 10. Interface para Exportação e Formatação Geométrica do Laudo Final
        caminho_saida = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Planilhas Excel", "*.xlsx")],
            title="Salvar Laudo Preditivo e Prognóstico",
            initialfile="Prognostico_Avancado_Banda_Total_6203ZZ.xlsx"
        )

        if not caminho_saida:
            return

        with pd.ExcelWriter(caminho_saida, engine='openpyxl') as writer:
            # Escrita dos tensores nas respectivas abas
            df_meta.to_excel(writer, sheet_name='Metadados Físicos', index=False)
            df_global.to_excel(writer, sheet_name='Métricas Globais', index=False)
            df_bandas.to_excel(writer, sheet_name='Energia por Bandas', index=False)
            df_harmonic.to_excel(writer, sheet_name='Rastreamento e Prognóstico', index=False)
            df_spec.to_excel(writer, sheet_name='Sinal Discretizado', index=False)

            # Algoritmo de AutoFit nativo via openpyxl
            for sheetname in writer.sheets:
                worksheet = writer.sheets[sheetname]
                for col in worksheet.columns:
                    max_length = 0
                    column_letter = col[0].column_letter # Coleta a referência alfabética da coluna (ex: 'A', 'B')
                    for cell in col:
                        try:
                            # Avalia o comprimento escalar da string formatada
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    # Aplica a transformação de largura com uma margem de tolerância visual (+2.5)
                    adjusted_width = (max_length + 2.5)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

        messagebox.showinfo(
            "Análise Concluída", 
            f"Processamento matricial espectral e formatação automática das matrizes finalizados.\nArquivo salvo em:\n{caminho_saida}"
        )

    except Exception as e:
        messagebox.showerror(
            "Erro Sistêmico", 
            f"Ocorreu uma falha na manipulação dos tensores ou formatação:\n{str(e)}\n\n"
            "Verifique se o arquivo selecionado contém as abas requeridas."
        )

if __name__ == "__main__":
    analisar_laudo_tecnico_conclusivo()