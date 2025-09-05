---
title: "Simulação de Monte Carlo — Futebol (Beta–Binomial / NegBin / Tempo até gol)"
emoji: ⚽🏃
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.44.1
python_version: "3.13"
app_file: app.py
pinned: false
short_description: Aplicações desenvolvidas no Mestrado Profissional da UNB

# Simulação de Monte Carlo — Palmeiras (2023–2025)

---
#Este Space implementa:

1. **Beta–Binomial** para taxa de conversão `p` (gols / chutes no alvo), com prior Uniforme, Jeffreys ou personalizada;  
2. **Negativa Binomial (mistura)** para estimar **quantos chutes no alvo (SOT)** são necessários para alcançar **k gols** com probabilidade alvo (ex.: 80% ou 90%);  
3. **Tempo até o gol** via **Exponencial** (Poisson thinning) e **Weibull** (shape `k`), usando uma taxa de SOT por minuto estimada de `SOT_total / (jogos * 90)`.

## Como usar
- Envie um CSV com colunas `gols` e `chutes_no_gol` (vírgula ou `;`). O app detecta e mapeia automaticamente.
- Ajuste prior, número de amostras, sementes e parâmetros (k, prob. alvo, jogos, minutos/jogo, shape k).
- Veja gráficos e tabelas; use como base para o relatório (PDF/HTML a partir do Jupyter/Colab).

 Observação: na aba **Tempo até gol** assumimos chegadas de SOT ~ Poisson com taxa aproximadamente constante. A taxa de gols é `λ_goal = p * λ_SOT`. Exponencial é o caso `k=1` da Weibull; `k≠1` flexibiliza risco crescente/decrescente.

## Estrutura
- `app.py` — interface Gradio (3 abas)
- `mc_core.py` — núcleo estatístico
- `requirements.txt` — dependências

## Execução local
```bash
pip install -r requirements.txt
python app.py
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
