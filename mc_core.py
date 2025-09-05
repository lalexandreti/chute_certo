import numpy as np
import pandas as pd
from typing import Tuple, Optional
from math import gamma  # escala da Weibull sem SciPy

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

def load_csv_any(path_or_buffer, sep_hint: Optional[str] = None) -> pd.DataFrame:
    """
    Aceita caminho (str) OU arquivo/bytes (ex.: st.file_uploader).
    Tenta vírgula; se detectar 1 coluna "entupida", relê com ';'.
    """
    # Caso 1: veio um objeto arquivo/bytes (tem .read)
    if hasattr(path_or_buffer, "read"):
        f = path_or_buffer
        # tentativa com vírgula
        f.seek(0)
        try:
            df = pd.read_csv(f)
        except Exception:
            # tentativa com ';'
            f.seek(0)
            df = pd.read_csv(f, sep=";")
        # fallback se ficou 1 coluna só
        if len(df.columns) == 1:
            f.seek(0)
            df = pd.read_csv(f, sep=";")
        return df

    # Caso 2: veio um caminho (str)
    if sep_hint is None:
        try:
            df = pd.read_csv(path_or_buffer)
        except Exception:
            df = pd.read_csv(path_or_buffer, sep=";")
        if len(df.columns) == 1:
            head_vals = df.head(3).iloc[:, 0].tolist()
            if (";" in df.columns[0]) or any(isinstance(x, str) and ";" in x for x in head_vals):
                df = pd.read_csv(path_or_buffer, sep=";")
        return df
    else:
        return pd.read_csv(path_or_buffer, sep=sep_hint)

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
# Negativa Binomial: SOT (trials) p/ atingir k gols
# ----------------------------
def negbin_trials_needed(p_samples: np.ndarray, k_successes: int, seed=2024) -> np.ndarray:
    if k_successes <= 0:
        raise ValueError("k_successes deve ser >= 1.")
    rng = np.random.default_rng(seed)
    failures = rng.negative_binomial(k_successes, p_samples)
    return failures + k_successes  # trials

def min_sot_for_k_prob(p_samples: np.ndarray, k_successes: int, target_prob: float, t_max: int = 3000) -> int:
    if not (0 < target_prob < 1):
        raise ValueError("target_prob deve estar entre (0,1).")
    T = k_successes
    while T <= t_max:
        rng = np.random.default_rng(12345 + T)
        goals_sim = rng.binomial(T, p_samples)
        if (goals_sim >= k_successes).mean() >= target_prob:
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
    if lambda_sot_per_min <= 0:
        raise ValueError("lambda_sot_per_min deve ser > 0.")
    rng = np.random.default_rng(seed)
    lambda_goal = np.clip(p_samples * lambda_sot_per_min, 1e-12, None)
    return rng.exponential(1.0 / lambda_goal, size=len(lambda_goal))

def time_to_goal_weibull(p_samples: np.ndarray, lambda_sot_per_min: float, shape_k: float = 1.5, n=100_000, seed=8) -> np.ndarray:
    if lambda_sot_per_min <= 0 or shape_k <= 0:
        raise ValueError("lambda_sot_per_min e shape_k devem ser > 0.")
    rng = np.random.default_rng(seed)
    lambda_goal = np.clip(p_samples * lambda_sot_per_min, 1e-12, None)
    theta = (1.0 / lambda_goal) / float(gamma(1.0 + 1.0/shape_k))  # mesma média do exponencial
    base = rng.weibull(shape_k, size=len(lambda_goal))  # scale=1
    return base * theta

def summarize(arr: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10":    float(np.percentile(arr, 10)),
        "p90":    float(np.percentile(arr, 90)),
    }
