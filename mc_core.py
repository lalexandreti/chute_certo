import io
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.special import gamma  # para escala da Weibull

# ----------------------------
# Loading / column mapping
# ----------------------------
ALIASES_GOALS = ["goals", "gols", "gls", "g"]
ALIASES_SOT   = ["shots_on_target", "chutes_no_gol", "finalizacoes_no_gol", "finalizações no alvo", "chutes no alvo"]

def _pick(colnames_lower, aliases):
    for a in aliases:
        if a.lower() in colnames_lower:
            return colnames_lower.index(a.lower())
    return None

def load_csv_any(path: str, sep_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Tenta ler com vírgula; se detectar 1 coluna "entupida" com ';', relê com ';'.
    """
    if sep_hint is None:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")
        if len(df.columns) == 1:
            head_vals = df.head(3).iloc[:, 0].tolist()
            if (";" in df.columns[0]) or any(isinstance(x, str) and ";" in x for x in head_vals):
                df = pd.read_csv(path, sep=";")
        return df
    else:
        return pd.read_csv(path, sep=sep_hint)

def map_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols_lower = [c.strip().lower() for c in df.columns]
    i_g = _pick(cols_lower, ALIASES_GOALS)
    i_s = _pick(cols_lower, ALIASES_SOT)
    # fuzzy fallback
    if i_g is None:
        for idx, c in enumerate(cols_lower):
            if "gol" in c and ("gols" in c or c == "g" or "goals" in c):
                i_g = idx; break
    if i_s is None:
        for idx, c in enumerate(cols_lower):
            if "chutes" in c and ("gol" in c or "alvo" in c):
                i_s = idx; break
        if i_s is None:
            for idx, c in enumerate(cols_lower):
                if "shots" in c and "target" in c:
                    i_s = idx; break
    if i_g is None or i_s is None:
        raise ValueError(f"Não achei colunas de gols/SOT. Colunas disponíveis: {df.columns.tolist()}")
    return df.columns[i_g], df.columns[i_s]

# ----------------------------
# Beta–Binomial posterior/predictive
# ----------------------------
def posterior_beta_params(goals: int, shots_on_target: int, alpha0=1.0, beta0=1.0):
    if shots_on_target < 0 or goals < 0 or goals > shots_on_target:
        raise ValueError("Valores inválidos para gols/SOT.")
    return (alpha0 + goals, beta0 + (shots_on_target - goals))

def sample_posterior_p(alpha_post: float, beta_post: float, n=100_000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.beta(alpha_post, beta_post, size=n)

def predictive_goals(p_samples: np.ndarray, s_star: int, seed=123):
    rng = np.random.default_rng(seed)
    return rng.binomial(int(s_star), p_samples)

# ----------------------------
# Negativa Binomial: SOT (trials) para atingir k gols
# ----------------------------
def negbin_trials_needed(p_samples: np.ndarray, k_successes: int, seed=2024) -> np.ndarray:
    """
    Para cada p ~ posterior, amostra 'falhas antes de k' ~ NegBin(k, p), depois trials = falhas + k.
    Retorna um array com SOT necessários (trials) para alcançar k gols.
    """
    if k_successes <= 0:
        raise ValueError("k_successes deve ser >= 1.")
    rng = np.random.default_rng(seed)
    failures = rng.negative_binomial(k_successes, p_samples)
    trials = failures + k_successes
    return trials

def min_sot_for_k_prob(p_samples: np.ndarray, k_successes: int, target_prob: float, t_max: int = 3000) -> int:
    """
    Menor T tal que E_p[ P(Binomial(T, p) >= k) ] >= target_prob, aproximado por simulação.
    """
    if not (0 < target_prob < 1):
        raise ValueError("target_prob deve estar entre (0,1).")
    T = k_successes
    while T <= t_max:
        rng = np.random.default_rng(12345 + T)
        goals_sim = rng.binomial(T, p_samples)
        prob = (goals_sim >= k_successes).mean()
        if prob >= target_prob:
            return T
        T += 1
    return t_max

# ----------------------------
# Tempo até o gol: Exponencial / Weibull
# ----------------------------
def estimate_lambda_sot_per_minute(sot_total: int, jogos: int = 38, minutos_por_jogo: int = 90) -> float:
    if jogos <= 0 or minutos_por_jogo <= 0:
        raise ValueError("jogos e minutos_por_jogo devem ser > 0.")
    return sot_total / (jogos * minutos_por_jogo)

def time_to_goal_exponential(p_samples: np.ndarray, lambda_sot_per_min: float, n=100_000, seed=7) -> np.ndarray:
    """
    Thinning de Poisson: lambda_goal = p * lambda_sot. Tempo ~ Exp(lambda_goal).
    """
    if lambda_sot_per_min <= 0:
        raise ValueError("lambda_sot_per_min deve ser > 0.")
    rng = np.random.default_rng(seed)
    lambda_goal = p_samples * lambda_sot_per_min
    lambda_goal = np.clip(lambda_goal, 1e-12, None)
    return rng.exponential(1.0 / lambda_goal, size=len(lambda_goal))

def time_to_goal_weibull(p_samples: np.ndarray, lambda_sot_per_min: float, shape_k: float = 1.5, n=100_000, seed=8) -> np.ndarray:
    """
    Weibull com shape k e scale ajustado para ter a mesma média do Exponencial (1/lambda_goal).
    Mean(Weibull(k, scale θ)) = θ * Γ(1 + 1/k). Então θ = (1/lambda_goal) / Γ(1 + 1/k).
    """
    if lambda_sot_per_min <= 0 or shape_k <= 0:
        raise ValueError("lambda_sot_per_min e shape_k devem ser > 0.")
    rng = np.random.default_rng(seed)
    lambda_goal = np.clip(p_samples * lambda_sot_per_min, 1e-12, None)
    theta = (1.0 / lambda_goal) / float(gamma(1.0 + 1.0/shape_k))
    base = rng.weibull(shape_k, size=len(lambda_goal))  # np: scale=1; ajustamos multiplicando por θ
    return base * theta

def summarize(arr: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p90":    float(np.percentile(arr, 90)),
    }
