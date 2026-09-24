"""
bnb_ml_oracle.py
================================================================
Oracle ML pour le marché "BNB Up-or-Down Daily" de predict.fun.

Objectif explicite (pas de recherche d'edge / d'indicateur technique) :
estimer P(BNB termine en hausse à 18h Paris vs la veille) à partir de
SEULEMENT trois quantités causales, rien d'autre :
  - l'écart de prix actuel par rapport à la référence de la veille (log_dev)
  - le temps restant avant la clôture (tau_hours) — +0.5% à 3h de la
    clôture n'a pas la même signification qu'à 23h
  - la volatilité EWMA de BNB/USDT, nécessaire pour convertir (écart +
    temps restant) en probabilité, au même titre que la vol implicite en
    pricing d'option — ce n'est pas un signal directionnel

Deux briques :
  1. Baseline ANALYTIQUE sans paramètre libre : probabilité "digitale"
     GBM driftless, P(up) = Phi(d) avec d = log_dev / (sigma_ewma * sqrt(tau)).
  2. Couche ML (régression logistique + gradient boosting) entraînée sur
     UNE SEULE feature dérivée — d_diffusion = ce même d standardisé — plus
     tau_hours séparément, pour laisser le ML corriger la forme Φ() si la
     fréquence empirique s'en écarte (queues plus épaisses, asymétrie...),
     sans jamais ajouter de RSI/momentum/order-flow. Évalué par walk-forward
     OOS contre ce baseline, pas contre un pile-ou-face à 50 %.

Mécanique du marché (identique à bitcoin-up-or-down-on-<date>, confirmée
par Matthieu) :
  - Référence figée au prix BNB/USDT à midi ET (= ~18h Paris hiver / 17h
    Paris été) le jour D-1.
  - Résolution 24h plus tard, à midi ET le jour D : UP si prix > référence.
  - Fenêtre de trading du bot : de 18h10 Paris (nouveau slug dispo) à
    15h Paris le lendemain (coupure avant résolution) → on ne modélise
    QUE cette fenêtre, jamais les 3h juste avant le fixing.

⚠️ Ce script ne peut pas être exécuté dans ce sandbox (api.binance.com
   est bloqué en sortie ici, 403). À lancer sur ton infra qui a l'accès
   Binance. Aucun chiffre de fiabilité ci-dessous n'est donc réel tant
   que tu ne l'as pas fait tourner toi-même sur l'historique réel.

Dépendances : pandas, numpy, scikit-learn, requests, scipy, joblib
    pip install pandas numpy scikit-learn requests scipy joblib
"""

from __future__ import annotations

import time
import math
import joblib
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

# ============================================================
# CONFIG
# ============================================================
SYMBOL = "BNBUSDT"
KLINES_INTERVAL = "1h"          # bougies horaires : assez fin pour capter
# la structure intra-cycle, assez léger
# pour couvrir plusieurs années
BINANCE_BASE = "https://api.binance.com"
ET_TZ = ZoneInfo("America/New_York")
PARIS_TZ = ZoneInfo("Europe/Paris")

BID_ASK_MARGIN = 0.01           # même convention que le bot (spread autour
# de la proba fair)

# Fenêtre de trading (doit matcher CRYPTO_DAILY_CUTOFF/RESTART du bot)
CUTOFF_HOUR_PARIS = 15
RESTART_HOUR_PARIS = 18
RESTART_MINUTE_PARIS = 10

# Pas d'échantillonnage des features à l'intérieur d'un cycle (backtest)
FEATURE_SAMPLE_STEP_HOURS = 1

RANDOM_STATE = 42


# ============================================================
# 1. RÉCUPÉRATION DES DONNÉES BINANCE (klines publiques)
# ============================================================

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Pagine sur /api/v3/klines (max 1000 bougies/requête).
    Retourne un DataFrame indexé par timestamp UTC avec OHLCV +
    taker_buy_base (proxy de pression acheteuse).
    """
    rows = []
    cursor = start_ms
    session = requests.Session()

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = session.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)  # rate limit courtoisie

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open", "high", "low", "close", "volume", "taker_buy_base", "n_trades"]]


def load_full_history(symbol: str = SYMBOL, interval: str = KLINES_INTERVAL,
                      years_back: int = 4) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years_back)
    print(f"📥 Téléchargement {symbol} {interval} de {start.date()} à {end.date()}...")
    df = fetch_binance_klines(symbol, interval, int(start.timestamp() * 1000), int(end.timestamp() * 1000))
    print(f"✅ {len(df)} bougies récupérées.")
    return df


# ============================================================
# 2. CONSTRUCTION DES CYCLES (référence J-1 midi ET → résolution J midi ET)
# ============================================================

def _next_noon_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Prochain midi ET strictement après ts (gère le passage DST via ZoneInfo)."""
    local = ts.tz_convert(ET_TZ)
    noon = local.replace(hour=12, minute=0, second=0, microsecond=0)
    if local >= noon:
        noon = noon + timedelta(days=1)
    return noon.tz_convert("UTC")


def price_at(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """Prix (close) de la bougie la plus proche <= ts (pas de look-ahead)."""
    sub = df.loc[:ts]
    if sub.empty:
        return None
    return float(sub["close"].iloc[-1])


@dataclass
class Cycle:
    cycle_id: int
    ref_time: pd.Timestamp
    ref_price: float
    res_time: pd.Timestamp
    res_price: float
    label: int  # 1 = up, 0 = down/flat


def build_cycles(df: pd.DataFrame) -> list[Cycle]:
    """Un cycle = un slug journalier bnb-up-or-down-on-<date>."""
    cycles = []
    t = df.index[0]
    ref_time = _next_noon_et(t)
    cid = 0
    while True:
        res_time = ref_time + timedelta(hours=24)
        if res_time > df.index[-1]:
            break
        ref_price = price_at(df, ref_time)
        res_price = price_at(df, res_time)
        if ref_price is None or res_price is None:
            ref_time = res_time
            continue
        cycles.append(Cycle(
            cycle_id=cid, ref_time=ref_time, ref_price=ref_price,
            res_time=res_time, res_price=res_price,
            label=int(res_price > ref_price),
        ))
        cid += 1
        ref_time = res_time
    print(f"📊 {len(cycles)} cycles journaliers construits "
          f"(base-rate up = {np.mean([c.label for c in cycles]):.3f})")
    return cycles


# ============================================================
# 3. FEATURES POINT-IN-TIME (calculées avec seulement des données <= t)
# ============================================================

def _log_ret(series: pd.Series, periods: int) -> pd.Series:
    return np.log(series / series.shift(periods))


def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Volontairement minimal : seule la volatilité EWMA est calculée (elle
    sert à convertir écart de prix + temps restant en probabilité — pas
    un indicateur directionnel). Aucun RSI/momentum/order-flow.
    """
    df = df.copy()
    df["ret_1h"] = _log_ret(df["close"], 1)
    df["sigma_ewma"] = df["ret_1h"].ewm(halflife=6).std() * math.sqrt(24)  # annualisée "par cycle" 24h
    return df


def in_trading_window(ts_utc: pd.Timestamp) -> bool:
    """Reproduit is_crypto_daily_trading_allowed() du bot (fuseau Paris)."""
    local = ts_utc.tz_convert(PARIS_TZ)
    h, m = local.hour, local.minute
    if h < CUTOFF_HOUR_PARIS:
        return True
    if h > RESTART_HOUR_PARIS:
        return True
    if h == RESTART_HOUR_PARIS and m >= RESTART_MINUTE_PARIS:
        return True
    return False


FEATURE_COLS = [
    "d_diffusion",  # log_dev / (sigma_ewma * sqrt(tau/24)) — écart standardisé
    "tau_hours",    # temps restant, gardé séparément pour laisser le ML
    # corriger la loi d'échelle sqrt(tau) si elle ne tient
    # pas exactement dans les données réelles
]


def build_feature_rows(df_ind: pd.DataFrame, cycles: list[Cycle]) -> pd.DataFrame:
    """
    Pour chaque cycle, échantillonne un point de features toutes les
    FEATURE_SAMPLE_STEP_HOURS à l'intérieur de la fenêtre de trading
    réellement utilisée par le bot (jamais dans les 3h avant fixing).
    Chaque ligne garde cycle_id pour un split walk-forward sans fuite
    (toutes les lignes d'un même cycle restent groupées).
    """
    rows = []
    for c in cycles:
        t = c.ref_time
        while t < c.res_time:
            if t in df_ind.index or True:  # tolérance : on cherche <= t
                if in_trading_window(t):
                    snap = df_ind.loc[:t]
                    if len(snap) < 30:
                        t += timedelta(hours=FEATURE_SAMPLE_STEP_HOURS)
                        continue
                    last = snap.iloc[-1]
                    S_t = last["close"]
                    tau_h = (c.res_time - t).total_seconds() / 3600.0
                    if tau_h <= 0:
                        t += timedelta(hours=FEATURE_SAMPLE_STEP_HOURS)
                        continue
                    sigma_ewma = last["sigma_ewma"]
                    log_dev = math.log(S_t / c.ref_price)
                    sigma_tau = sigma_ewma * math.sqrt(tau_h / 24.0) if sigma_ewma and sigma_ewma > 0 else None
                    if not sigma_tau or sigma_tau <= 0:
                        t += timedelta(hours=FEATURE_SAMPLE_STEP_HOURS)
                        continue
                    rows.append({
                        "cycle_id": c.cycle_id,
                        "t": t,
                        "log_dev": log_dev,               # gardé pour diagnostic/baseline, pas dans FEATURE_COLS
                        "tau_hours": tau_h,
                        "sigma_ewma": sigma_ewma,          # gardé pour diagnostic/baseline
                        "d_diffusion": log_dev / sigma_tau,
                        "sigma_for_gbm": sigma_ewma,
                        "label": c.label,
                    })
            t += timedelta(hours=FEATURE_SAMPLE_STEP_HOURS)

    feat = pd.DataFrame(rows).dropna(subset=FEATURE_COLS + ["label"])
    print(f"🧮 {len(feat)} lignes de features sur {feat['cycle_id'].nunique()} cycles.")
    return feat


# ============================================================
# 4. BASELINE ANALYTIQUE — digitale GBM driftless
# ============================================================

def analytic_prob_up(S_t: float, K: float, tau_hours: float, sigma_annualized_proxy: float) -> float:
    """
    P(S_T > K) sous GBM sans dérive (marché efficient) :
        d = log(S_t/K) / (sigma * sqrt(tau))
        P(up) = Phi(d)
    sigma_annualized_proxy = écart-type horaire * sqrt(24) déjà annualisé
    "par cycle" (cf. sigma_ewma ci-dessus) ; on le remet à l'échelle de tau.
    """
    if sigma_annualized_proxy <= 0 or tau_hours <= 0:
        return 0.5
    sigma_tau = sigma_annualized_proxy * math.sqrt(tau_hours / 24.0)
    if sigma_tau <= 0:
        return 0.5
    d = math.log(S_t / K) / sigma_tau
    return float(norm.cdf(d))


# ============================================================
# 5. WALK-FORWARD BACKTEST (groupé par cycle_id, jamais par ligne)
# ============================================================

def walk_forward_splits(cycle_ids: np.ndarray, n_folds: int = 8, min_train_frac: float = 0.3):
    """Découpe les cycles (triés chronologiquement) en folds expanding-window."""
    uniq = np.sort(np.unique(cycle_ids))
    n = len(uniq)
    start = int(n * min_train_frac)
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        train_cycles = uniq[:fold_edges[i]]
        test_cycles = uniq[fold_edges[i]:fold_edges[i + 1]]
        if len(test_cycles) == 0:
            continue
        yield train_cycles, test_cycles


def collect_oos_predictions(feat: pd.DataFrame, n_folds: int = 8) -> pd.DataFrame:
    """
    Même boucle walk-forward que précédemment, mais retourne les
    prédictions LIGNE PAR LIGNE (pas juste des moyennes agrégées), pour
    permettre ensuite : ventilation par horizon, calibration, test de
    significativité clusterisé par cycle.
    """
    frames = []

    for train_cycles, test_cycles in walk_forward_splits(feat["cycle_id"].values, n_folds):
        train = feat[feat["cycle_id"].isin(train_cycles)]
        test = feat[feat["cycle_id"].isin(test_cycles)].copy()
        if train.empty or test.empty:
            continue

        X_train, y_train = train[FEATURE_COLS].values, train["label"].values
        X_test = test[FEATURE_COLS].values

        scaler = StandardScaler().fit(X_train)
        Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

        test["p_naive_50"] = 0.5
        test["p_prior_hist"] = train["label"].mean()
        test["p_gbm_analytique"] = test.apply(
            lambda r: analytic_prob_up(math.exp(r["log_dev"]), 1.0, r["tau_hours"], r["sigma_for_gbm"]),
            axis=1,
        )

        logit = LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE)
        logit.fit(Xtr, y_train)
        test["p_logit_ml"] = logit.predict_proba(Xte)[:, 1]

        gbc = GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.03,
            subsample=0.8, random_state=RANDOM_STATE,
        )
        gbc.fit(Xtr, y_train)
        test["p_gbc_ml"] = gbc.predict_proba(Xte)[:, 1]

        frames.append(test)

    return pd.concat(frames, ignore_index=True)


PRED_COLS = ["p_naive_50", "p_prior_hist", "p_gbm_analytique", "p_logit_ml", "p_gbc_ml"]


def summarize_predictions(oos: pd.DataFrame) -> pd.DataFrame:
    """Reproduit le tableau agrégé d'origine (Brier/logloss/accuracy par modèle)."""
    rows = []
    y = oos["label"].values
    for col in PRED_COLS:
        p = np.clip(oos[col].values, 1e-4, 1 - 1e-4)
        rows.append({
            "model": col.replace("p_", ""),
            "brier_mean": brier_score_loss(y, p),
            "logloss_mean": log_loss(y, p),
            "accuracy_mean": accuracy_score(y, (p >= 0.5).astype(int)),
            "n_total": len(y),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("brier_mean")


def backtest_by_horizon(oos: pd.DataFrame, bucket_edges=(0, 3, 6, 9, 12, 15, 18, 24)) -> pd.DataFrame:
    """
    Brier/accuracy par tranche de temps-restant. Essentiel : le Brier
    pooled est dominé par les échantillons proches de la clôture, où la
    prédiction devient quasi triviale (d diverge quand tau -> 0). Ce qui
    compte pour le market making, c'est la qualité tôt dans la fenêtre.
    """
    oos = oos.copy()
    oos["tau_bucket"] = pd.cut(oos["tau_hours"], bins=bucket_edges)
    rows = []
    for bucket, g in oos.groupby("tau_bucket", observed=True):
        if g.empty:
            continue
        y = g["label"].values
        for col in PRED_COLS:
            p = np.clip(g[col].values, 1e-4, 1 - 1e-4)
            rows.append({
                "tau_bucket": str(bucket),
                "model": col.replace("p_", ""),
                "brier": brier_score_loss(y, p),
                "accuracy": accuracy_score(y, (p >= 0.5).astype(int)),
                "n": len(y),
            })
    return pd.DataFrame(rows).pivot(index="tau_bucket", columns="model", values="brier")


def calibration_table(oos: pd.DataFrame, model_col: str = "p_gbm_analytique", n_bins: int = 10) -> pd.DataFrame:
    """
    Proba prédite (par décile) vs fréquence réalisée. Un modèle bien
    calibré doit avoir realized_freq ≈ mean_predicted dans chaque bin —
    c'est ce qui compte pour fixer un bid/ask, pas juste l'accuracy.
    """
    oos = oos.copy()
    oos["bin"] = pd.qcut(oos[model_col], n_bins, duplicates="drop")
    tbl = oos.groupby("bin", observed=True).agg(
        mean_predicted=(model_col, "mean"),
        realized_freq=("label", "mean"),
        n=("label", "size"),
    )
    return tbl


def cluster_significance_test(oos: pd.DataFrame, model_col: str, baseline_col: str) -> dict:
    """
    Test apparié sur la différence de perte quadratique, CLUSTERISÉ par
    cycle_id : on moyenne d'abord la différence de perte au sein de
    chaque cycle (les ~20 lignes d'un même cycle ne comptent que pour un
    point), puis t-test sur ces moyennes par cycle. Contrairement à un
    test ligne-par-ligne, ceci ne pseudo-réplique pas les observations
    corrélées intra-cycle et ne gonfle donc pas artificiellement la
    significativité.
    """
    y = oos["label"].values
    d = (oos[model_col].values - y) ** 2 - (oos[baseline_col].values - y) ** 2
    per_cycle = pd.Series(d, index=oos["cycle_id"].values).groupby(level=0).mean()
    mean_d = per_cycle.mean()
    se_d = per_cycle.std(ddof=1) / math.sqrt(len(per_cycle))
    t_stat = mean_d / se_d if se_d > 0 else np.nan
    return {
        "mean_diff_brier": mean_d,  # négatif = model_col meilleur que baseline_col
        "t_stat": t_stat,
        "n_cycles": len(per_cycle),
        "significant_5pct": bool(abs(t_stat) > 1.96) if not np.isnan(t_stat) else None,
    }


# ============================================================
# 6. MODÈLE FINAL (entraîné sur tout l'historique) + INFÉRENCE LIVE
# ============================================================

def train_final_model(feat: pd.DataFrame, out_path: str = "bnb_oracle_model.joblib"):
    X = feat[FEATURE_COLS].values
    y = feat["label"].values
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, C=0.5, random_state=RANDOM_STATE)
    model.fit(scaler.transform(X), y)
    joblib.dump({"model": model, "scaler": scaler, "features": FEATURE_COLS}, out_path)
    print(f"💾 Modèle sauvegardé → {out_path}")
    return model, scaler


def predict_up_probability(model, scaler, S_t: float, K: float, tau_hours: float,
                           sigma_ewma: float, blend_with_gbm: float = 0.5) -> float:
    """
    Inférence live. Entrées : prix courant, référence de la veille, temps
    restant, volatilité EWMA — rien d'autre.
    blend_with_gbm ∈ [0,1] : pondère la sortie ML avec le baseline
    analytique (recommandé tant que le ML n'a pas prouvé un edge net et
    stable en walk-forward — voir run_backtest()).
    """
    log_dev = math.log(S_t / K)
    sigma_tau = sigma_ewma * math.sqrt(tau_hours / 24.0) if sigma_ewma and sigma_ewma > 0 else None
    if not sigma_tau or sigma_tau <= 0:
        return 0.5
    d = log_dev / sigma_tau
    x = np.array([[d, tau_hours]])
    p_ml = float(model.predict_proba(scaler.transform(x))[:, 1][0])
    p_gbm = analytic_prob_up(S_t, K, tau_hours, sigma_ewma)
    return blend_with_gbm * p_gbm + (1 - blend_with_gbm) * p_ml


# ============================================================
# 7. ADAPTATEUR ORACLE — même format de sortie que get_polymarket_par_slug()
#    (à appeler depuis le bot pour le slug BNB daily uniquement)
# ============================================================

def get_bnb_ml_oracle(slug: str, ref_price: float, ref_time: pd.Timestamp,
                      model, scaler, current_kline_1h_tail: pd.DataFrame,
                      question_up: str, question_down: str) -> list[dict]:
    """
    Reproduit exactement le format renvoyé par get_polymarket_par_slug()
    (liste de dicts avec question/best_bid/best_ask), pour brancher sans
    toucher à should_remove_orders()/get_buy_signal() côté bot : il suffit
    de router process_market(marche_bnb_daily, ...) sur cette fonction au
    lieu de get_polymarket_par_slug().

    current_kline_1h_tail : DataFrame des dernières ~48h de bougies 1h,
    déjà enrichi via enrich_indicators(), avec le prix courant en dernière ligne.
    """
    now = current_kline_1h_tail.index[-1]
    last = current_kline_1h_tail.iloc[-1]
    S_t = last["close"]
    tau_hours = (ref_time + timedelta(hours=24) - now).total_seconds() / 3600.0

    p_up = predict_up_probability(
        model, scaler, S_t, ref_price, tau_hours, last["sigma_ewma"],
    )
    p_down = 1.0 - p_up

    return [
        {"question": question_up, "token_id": None,
         "best_bid": max(0.0, p_up - BID_ASK_MARGIN),
         "best_ask": min(1.0, p_up + BID_ASK_MARGIN),
         "spread": None, "group_item_title": None, "market_slug": slug},
        {"question": question_down, "token_id": None,
         "best_bid": max(0.0, p_down - BID_ASK_MARGIN),
         "best_ask": min(1.0, p_down + BID_ASK_MARGIN),
         "spread": None, "group_item_title": None, "market_slug": slug},
    ]


# ============================================================
# MAIN — à lancer sur ton infra (accès Binance requis)
# ============================================================

def main():
    df = load_full_history(years_back=4)
    df_ind = enrich_indicators(df)
    cycles = build_cycles(df_ind)
    feat = build_feature_rows(df_ind, cycles)

    print("\n=== Backtest walk-forward (expanding window, 8 folds) ===")
    oos = collect_oos_predictions(feat, n_folds=8)
    print(summarize_predictions(oos).to_string())

    print("\n=== Ventilation par temps restant avant clôture (Brier) ===")
    print(backtest_by_horizon(oos).to_string())
    print("💡 Si gbm_analytique reste proche de logit_ml/gbc_ml même sur les")
    print("   tranches tau élevées (>12h, la partie difficile et la plus")
    print("   utile pour le market making), c'est un signe fort de fiabilité")
    print("   du baseline analytique — et donc de l'inutilité du ML.")

    print("\n=== Calibration du baseline analytique (déciles) ===")
    print(calibration_table(oos, "p_gbm_analytique").to_string())
    print("💡 mean_predicted doit être proche de realized_freq dans chaque bin.")

    print("\n=== Significativité logit_ml vs gbm_analytique (clusterisé par cycle) ===")
    test_result = cluster_significance_test(oos, "p_logit_ml", "p_gbm_analytique")
    print(test_result)
    if test_result["significant_5pct"]:
        print("→ Écart statistiquement significatif au seuil 5%.")
    else:
        print("→ PAS significatif : l'écart observé est probablement du bruit "
              "d'échantillonnage, pas un vrai edge du ML.")

    # Entraînement final sur tout l'historique pour la mise en prod
    # (à ne déployer que si le test ci-dessus est significatif ET stable
    #  dans le temps — sinon, utiliser directement analytic_prob_up()
    #  sans modèle du tout : plus simple, plus robuste, zéro risque
    #  d'overfitting, et ça ne demande aucun réentraînement périodique).
    model, scaler = train_final_model(feat)


if __name__ == "__main__":
    main()
