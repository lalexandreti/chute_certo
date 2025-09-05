import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem GUI
import matplotlib.pyplot as plt
import streamlit as st

from mc_core import (
    load_csv_any, map_columns, posterior_beta_params, sample_posterior_p, predictive_goals,
    negbin_trials_needed, min_sot_for_k_prob,
    estimate_lambda_sot_per_minute, time_to_goal_exponential, time_to_goal_weibull, summarize
)

# ------------------ Config página ------------------
st.set_page_config(page_title="Simulação de Monte Carlo — Futebol", layout="wide")

# Cabeçalho com logos (se existirem no repo)
left, mid, right = st.columns([1, 6, 1])
with left:
    try:
        st.image("MARCADOR.png", width=120)  # trocado de use_container_width -> width
    except Exception:
        pass
with mid:
    st.markdown("<h1 style='text-align:center;color:#003366;'>Análise de Distribuições de Probabilidade</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#003366;'>Mestrando Luiz Alexandre Rodrigues Silva</h3>", unsafe_allow_html=True)
with right:
    try:
        st.image("MARCADOR.png", width=120)
    except Exception:
        pass

st.markdown("---")

# Helpers
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

# UI: abas
tab1, tab2, tab3 = st.tabs([
    "Beta–Binomial (taxa de conversão)",
    "Negativa Binomial (SOT para k gols)",
    "Tempo até gol (Exponencial / Weibull)"
])

# ------------------ Aba 1: Beta–Binomial ------------------
with tab1:
    st.subheader("Posterior de p e preditiva de gols")
    uploaded = st.file_uploader("CSV (opcional) — vírgula ou ';' — colunas: gols, chutes_no_gol", type=["csv"])
    prior = st.radio("Prior para p", ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha0 = st.number_input("α0 (se personalizada)", value=1.0)
    with c2:
        beta0 = st.number_input("β0 (se personalizada)", value=1.0)
    with c3:
        S_star = st.slider("S* (futuros SOT)", 1, 60, 30, 1)
    with c4:
        draws = st.slider("Amostras Monte Carlo", 20000, 300000, 120000, 10000)

    c5, c6 = st.columns(2)
    with c5:
        seed_post = st.number_input("Semente posterior", value=42, step=1)
    with c6:
        seed_pred = st.number_input("Semente preditiva", value=123, step=1)

    if st.button("Rodar (Beta–Binomial)"):
        df = load_csv_any(uploaded) if uploaded else _fallback_df()

        col_g, col_sot = map_columns(df)
        df = _ensure_numeric(df, col_g, col_sot)
        G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

        if prior.startswith("Uniforme"):
            a0, b0 = 1.0, 1.0
        elif prior.startswith("Jeffreys"):
            a0, b0 = 0.5, 0.5
        else:
            a0, b0 = float(alpha0), float(beta0)

        a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
        p_samples = sample_posterior_p(a_post, b_post, n=int(draws), seed=int(seed_post))
        pred = predictive_goals(p_samples, int(S_star), seed=int(seed_pred))

        st.write(pd.DataFrame([
            {"metric": "p (taxa de conversão)", **summarize(p_samples)},
            {"metric": f"gols em {int(S_star)} SOT", **summarize(pred)},
        ]))

        # Gráficos
        fig1 = plt.figure()
        plt.hist(p_samples, bins=60, density=True)
        plt.title("Posterior de p"); plt.xlabel("p"); plt.ylabel("densidade"); plt.tight_layout()
        st.pyplot(fig1)

        fig2 = plt.figure()
        bins = range(int(np.min(pred)), int(np.max(pred))+2)
        plt.hist(pred, bins=bins, density=True)
        plt.title(f"Distribuição preditiva: gols em {int(S_star)} SOT")
        plt.xlabel("gols"); plt.ylabel("densidade"); plt.tight_layout()
        st.pyplot(fig2)

# ------------------ Aba 2: Negativa Binomial ------------------
with tab2:
    st.subheader("SOT necessários para atingir k gols com alta probabilidade")
    uploaded2 = st.file_uploader("CSV (opcional)", type=["csv"], key="csv2")
    prior2 = st.radio("Prior para p", ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"], horizontal=True, key="prior2")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha02 = st.number_input("α0 (se personalizada)", value=1.0, key="a02")
    with c2:
        beta02  = st.number_input("β0 (se personalizada)", value=1.0, key="b02")
    with c3:
        k_goals = st.slider("k (gols alvo)", 1, 20, 5, 1)
    with c4:
        target_prob = st.slider("Probabilidade alvo P(G≥k)", 0.50, 0.99, 0.80, 0.01)

    c5, c6 = st.columns(2)
    with c5:
        draws2 = st.slider("Amostras Monte Carlo", 20000, 300000, 120000, 10000, key="draws2")
    with c6:
        seed_post2 = st.number_input("Semente posterior", value=42, step=1, key="seedp2")

    if st.button("Rodar (NegBin)"):
        df = load_csv_any(uploaded2) if uploaded2 else _fallback_df()
        col_g, col_sot = map_columns(df)
        df = _ensure_numeric(df, col_g, col_sot)
        G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

        if prior2.startswith("Uniforme"):
            a0, b0 = 1.0, 1.0
        elif prior2.startswith("Jeffreys"):
            a0, b0 = 0.5, 0.5
        else:
            a0, b0 = float(alpha02), float(beta02)

        a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
        p_samples = sample_posterior_p(a_post, b_post, n=int(draws2), seed=int(seed_post2))

        trials = negbin_trials_needed(p_samples, int(k_goals), seed=2024)
        T_min = min_sot_for_k_prob(p_samples, int(k_goals), float(target_prob), t_max=3000)

        st.write(pd.DataFrame([{"metric": f"SOT p/ {int(k_goals)} gols", **summarize(trials)}]))
        st.info(f"Min SOT para P(G≥{int(k_goals)})≥{target_prob:.0%}: **{T_min}**")

        fig = plt.figure()
        plt.hist(trials, bins=60, density=True)
        plt.title(f"SOT necessários para {int(k_goals)} gols")
        plt.xlabel("SOT"); plt.ylabel("densidade"); plt.tight_layout()
        # marcadores visuais
        ymax = plt.ylim()[1]
        plt.axvline(T_min); plt.text(T_min, ymax*0.9, f"Min T≈{T_min}", rotation=90, va="top")
        st.pyplot(fig)

# ------------------ Aba 3: Tempo até gol ------------------
with tab3:
    st.subheader("Tempo até o gol — Exponencial / Weibull")
    uploaded3 = st.file_uploader("CSV (opcional)", type=["csv"], key="csv3")
    prior3 = st.radio("Prior para p", ["Uniforme (α=1, β=1)", "Jeffreys (α=0.5, β=0.5)", "Personalizada"], horizontal=True, key="prior3")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha03 = st.number_input("α0 (se personalizada)", value=1.0, key="a03")
    with c2:
        beta03  = st.number_input("β0 (se personalizada)", value=1.0, key="b03")
    with c3:
        jogos = st.number_input("Jogos", value=38, step=1)
    with c4:
        min_por_jogo = st.number_input("Minutos por jogo", value=90, step=1)

    c5, c6, c7 = st.columns(3)
    with c5:
        weibull_k = st.number_input("Weibull shape k (>0)", value=1.5)
    with c6:
        draws3 = st.slider("Amostras Monte Carlo", 20000, 300000, 120000, 10000, key="draws3")
    with c7:
        seed_post3 = st.number_input("Semente posterior", value=42, step=1, key="seedp3")

    if st.button("Rodar (Tempo até gol)"):
        df = load_csv_any(uploaded3) if uploaded3 else _fallback_df()
        col_g, col_sot = map_columns(df)
        df = _ensure_numeric(df, col_g, col_sot)
        G_total, SOT_total = int(df[col_g].fillna(0).sum()), int(df[col_sot].fillna(0).sum())

        if prior3.startswith("Uniforme"):
            a0, b0 = 1.0, 1.0
        elif prior3.startswith("Jeffreys"):
            a0, b0 = 0.5, 0.5
        else:
            a0, b0 = float(alpha03), float(beta03)

        a_post, b_post = posterior_beta_params(G_total, SOT_total, a0, b0)
        p_samples = sample_posterior_p(a_post, b_post, n=int(draws3), seed=int(seed_post3))

        lam_sot = estimate_lambda_sot_per_minute(SOT_total, int(jogos), int(min_por_jogo))
        t_exp = time_to_goal_exponential(p_samples, lam_sot, n=int(draws3), seed=7)
        t_wei = time_to_goal_weibull(p_samples, lam_sot, shape_k=float(weibull_k), n=int(draws3), seed=8)

        st.write(pd.DataFrame([
            {"metric": "Tempo até gol (Exponencial, min)", **summarize(t_exp)},
            {"metric": f"Tempo até gol (Weibull k={weibull_k}, min)", **summarize(t_wei)},
        ]))

        fig1 = plt.figure()
        plt.hist(t_exp, bins=60, density=True)
        plt.title("Tempo até gol — Exponencial"); plt.xlabel("min"); plt.ylabel("densidade"); plt.tight_layout()
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.hist(t_wei, bins=60, density=True)
        plt.title(f"Tempo até gol — Weibull (k={weibull_k})"); plt.xlabel("min"); plt.ylabel("densidade"); plt.tight_layout()
        st.pyplot(fig2)

