import io
import numpy as np
import pandas as pd
import matplotlib matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from mc_core import (
    load_csv_any, map_columns, posterior_beta_params, sample_posterior_p, predictive_goals,
    negbin_trials_needed, min_sot_for_k_prob,
    estimate_lambda_sot_per_minute, time_to_goal_exponential, time_to_goal_weibull, summarize
)

# -------- helpers --------
def _ensure_numeric(df, col_g, col_sot):
    df[col_g] = pd.to_numeric(df[col_g], errors="coerce")
    df[col_sot] = pd.to_numeric(df[col_sot], errors="coerce")
    return df

def _fallback_df():
    csv_text = (
        "temporada,total_de_chutes,chutes_no_gol,gols,defendidos,chutes_para_fora,bloqueados,na_trave\n"
        "2023,800,276,84,192,400,124,12\n"
        "2024,778,253,75,178,283,242,13\n"
        "2025,473,150,53,97,208,115,6\n"
    )
    return pd.read_csv(io.StringIO(csv_text))

# -------- tab 1: Beta–Binomial --------
def run_beta_binomial(file, prior_kind, alpha0, beta0, S_star, draws, seed_post, seed_pred):
    df = load_csv_any(file.name) if file is not None else _fallback_df()
    col_g, col_sot = map_columns(df)
    df = _ensure_numeric(df, col_g, col_sot)
    G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

    if prior_kind == "Uniforme (α=1, β=1)":
        a0, b0 = 1.0, 1.0
    elif prior_kind == "Jeffreys (α=0.5, β=0.5)":
        a0, b0 = 0.5, 0.5
    else:
        a0, b0 = float(alpha0), float(beta0)

    a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
    p_samples = sample_posterior_p(a_post, b_post, n=int(draws), seed=int(seed_post))

    pred = predictive_goals(p_samples, int(S_star), seed=int(seed_pred))
    table = pd.DataFrame([
        {"metric": "p (taxa de conversão)", **summarize(p_samples)},
        {"metric": f"gols em {int(S_star)} SOT", **summarize(pred)},
    ])

    fig1 = plt.figure()
    plt.hist(p_samples, bins=60, density=True)
    plt.title("Posterior de p"); plt.xlabel("p"); plt.ylabel("densidade"); plt.tight_layout()

    fig2 = plt.figure()
    bins = range(int(np.min(pred)), int(np.max(pred)) + 2)
    plt.hist(pred, bins=bins, density=True)
    plt.title(f"Distribuição preditiva: gols em {int(S_star)} SOT")
    plt.xlabel("gols"); plt.ylabel("densidade"); plt.tight_layout()

    meta = f"G_total={G_total}, SOT_total={SOT_total}\nPosterior: alpha={a_post:.3f}, beta={b_post:.3f}"
    return meta, table, fig1, fig2

# -------- tab 2: Negativa Binomial --------
def run_negbin(file, prior_kind, alpha0, beta0, k_goals, target_prob, draws, seed_post):
    df = load_csv_any(file.name) if file is not None else _fallback_df()
    col_g, col_sot = map_columns(df)
    df = _ensure_numeric(df, col_g, col_sot)
    G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

    if prior_kind == "Uniforme (α=1, β=1)":
        a0, b0 = 1.0, 1.0
    elif prior_kind == "Jeffreys (α=0.5, β=0.5)":
        a0, b0 = 0.5, 0.5
    else:
        a0, b0 = float(alpha0), float(beta0)

    a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
    p_samples = sample_posterior_p(a_post, b_post, n=int(draws), seed=int(seed_post))

    trials = negbin_trials_needed(p_samples, int(k_goals), seed=2024)
    table = pd.DataFrame([{"metric": f"SOT p/ {int(k_goals)} gols", **summarize(trials)}])

    T_min = min_sot_for_k_prob(p_samples, int(k_goals), float(target_prob), t_max=3000)
    meta = f"G_total={G_total}, SOT_total={SOT_total}\nMin SOT p/ P(G>={int(k_goals)})>={float(target_prob):.0%}: {T_min}"

    fig = plt.figure()
    plt.hist(trials, bins=60, density=True)
    plt.title(f"SOT necessários para {int(k_goals)} gols")
    plt.xlabel("SOT"); plt.ylabel("densidade"); plt.tight_layout()

    return meta, table, fig

# -------- tab 3: Tempo até o gol --------
def run_time_to_goal(file, prior_kind, alpha0, beta0, jogos, min_por_jogo, weibull_k, draws, seed_post):
    df = load_csv_any(file.name) if file is not None else _fallback_df()
    col_g, col_sot = map_columns(df)
    df = _ensure_numeric(df, col_g, col_sot)
    G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

    if prior_kind == "Uniforme (α=1, β=1)":
        a0, b0 = 1.0, 1.0
    elif prior_kind == "Jeffreys (α=0.5, β=0.5)":
        a0, b0 = 0.5, 0.5
    else:
        a0, b0 = float(alpha0), float(beta0)

    a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
    p_samples = sample_posterior_p(a_post, b_post, n=int(draws), seed=int(seed_post))

    lam_sot = estimate_lambda_sot_per_minute(SOT_total, int(jogos), int(min_por_jogo))
    t_exp = time_to_goal_exponential(p_samples, lam_sot, n=int(draws), seed=7)
    t_wei = time_to_goal_weibull(p_samples, lam_sot, shape_k=float(weibull_k), n=int(draws), seed=8)

    tab = pd.DataFrame([
        {"metric": "Tempo até gol (Exponencial, min)", **summarize(t_exp)},
        {"metric": f"Tempo até gol (Weibull k={weibull_k}, min)", **summarize(t_wei)},
    ])

    fig1 = plt.figure()
    plt.hist(t_exp, bins=60, density=True)
    plt.title("Tempo até gol — Exponencial"); plt.xlabel("min"); plt.ylabel("densidade"); plt.tight_layout()

    fig2 = plt.figure()
    plt.hist(t_wei, bins=60, density=True)
    plt.title(f"Tempo até gol — Weibull (k={weibull_k})"); plt.xlabel("min"); plt.ylabel("densidade"); plt.tight_layout()

    meta = f"SOT_total={SOT_total}, λ_SOT={lam_sot:.4f} por minuto"
    return meta, tab, fig1, fig2

# -------- UI --------
with gr.Blocks(title="Simulação de Monte Carlo — Futebol (Beta–Binomial / NegBin / Tempo até gol)") as demo:
    gr.Markdown("## ⚽ Simulação de Monte Carlo — Palmeiras 2023–2025")

    with gr.Tabs():
        with gr.TabItem("Beta–Binomial (taxa de conversão)"):
            file_in = gr.File(label="CSV (opcional)")
            prior_kind = gr.Radio(
                ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"],
                value="Uniforme (α=1, β=1)"
            )
            alpha0 = gr.Number(value=1.0, label="α0 (se personalizada)")
            beta0  = gr.Number(value=1.0, label="β0 (se personalizada)")
            S_star = gr.Slider(1, 60, value=30, step=1, label="S* (futuros SOT)")
            draws  = gr.Slider(20_000, 300_000, value=120_000, step=10_000, label="Amostras Monte Carlo")
            seed_p = gr.Number(value=42, label="Semente posterior")
            seed_g = gr.Number(value=123, label="Semente preditiva")
            btn = gr.Button("Rodar")

            meta_out   = gr.Textbox(label="Resumo")
            table_out  = gr.Dataframe(label="Estatísticas", interactive=False)
            plot_post  = gr.Plot(label="Posterior de p")
            plot_pred  = gr.Plot(label="Preditiva de gols")

            btn.click(run_beta_binomial,
                      [file_in, prior_kind, alpha0, beta0, S_star, draws, seed_p, seed_g],
                      [meta_out, table_out, plot_post, plot_pred])

        with gr.TabItem("Negativa Binomial (SOT para k gols)"):
            file_in2 = gr.File(label="CSV (opcional)")
            prior_kind2 = gr.Radio(
                ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"],
                value="Uniforme (α=1, β=1)"
            )
            alpha02 = gr.Number(value=1.0, label="α0 (se personalizada)")
            beta02  = gr.Number(value=1.0, label="β0 (se personalizada)")
            k_goals = gr.Slider(1, 20, value=5, step=1, label="k (gols alvo)")
            target_prob = gr.Slider(0.5, 0.99, value=0.8, step=0.01, label="Probabilidade alvo P(G≥k)")
            draws2  = gr.Slider(20_000, 300_000, value=120_000, step=10_000, label="Amostras Monte Carlo")
            seed_p2 = gr.Number(value=42, label="Semente posterior")
            btn2 = gr.Button("Rodar")

            meta2 = gr.Textbox(label="Resumo")
            table2 = gr.Dataframe(label="Estatísticas", interactive=False)
            plot2  = gr.Plot(label="SOT necessários (distribuição)")

            btn2.click(run_negbin,
                       [file_in2, prior_kind2, alpha02, beta02, k_goals, target_prob, draws2, seed_p2],
                       [meta2, table2, plot2])

        with gr.TabItem("Tempo até gol (Exponencial / Weibull)"):
            file_in3 = gr.File(label="CSV (opcional)")
            prior_kind3 = gr.Radio(
                ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"],
                value="Uniforme (α=1, β=1)"
            )
            alpha03 = gr.Number(value=1.0, label="α0 (se personalizada)")
            beta03  = gr.Number(value=1.0, label="β0 (se personalizada)")
            jogos = gr.Number(value=38, label="Jogos no dataset")
            min_por_jogo = gr.Number(value=90, label="Minutos por jogo")
            weibull_k = gr.Number(value=1.5, label="Weibull shape k (>0)")
            draws3  = gr.Slider(20_000, 300_000, value=120_000, step=10_000, label="Amostras Monte Carlo")
            seed_p3 = gr.Number(value=42, label="Semente posterior")
            btn3 = gr.Button("Rodar")

            meta3 = gr.Textbox(label="Resumo")
            table3 = gr.Dataframe(label="Estatísticas", interactive=False)
            plot3a = gr.Plot(label="Tempo até gol — Exponencial")
            plot3b = gr.Plot(label="Tempo até gol — Weibull")

            btn3.click(run_time_to_goal,
                       [file_in3, prior_kind3, alpha03, beta03, jogos, min_por_jogo, weibull_k, draws3, seed_p3],
                       [meta3, table3, plot3a, plot3b])

if __name__ == "__main__":
    demo.launch()
