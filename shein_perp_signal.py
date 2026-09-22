"""
Modèle de probabilité de hausse/baisse de SHEIN (0625.HK) à J+1 (prochaine clôture HKEX),
basé sur le rendement du perpetual xyz:SHEIN sur Hyperliquid depuis la dernière clôture.

Idée : le perp trade 24/7 pendant que HKEX est fermé (nuits, week-ends). Son évolution
pendant cette fenêtre est traitée comme un signal probabiliste avancé (exactement comme un
future overnight sur indice), avec une pondération par le temps restant avant réouverture,
sur le modèle d'un calcul de probabilité de franchissement (type N(d2) en pricing d'option) :

    z = (gamma0 + gamma1 * r / sqrt(tau))
    P(hausse) = Phi(z)

    r   = rendement cumulé du perp depuis la dernière clôture HKEX
    tau = temps restant avant la prochaine clôture HKEX (en heures, ou toute unité cohérente)

gamma0, gamma1 sont calibrés par régression probit sur historique (pas fixés a priori).

Dépendances : pip install requests pandas numpy scipy statsmodels yfinance --break-system-packages

------------------------------------------------------------------------------------------
IMPORTANT SUR LES DONNÉES
------------------------------------------------------------------------------------------
Ce script suppose que tu peux atteindre :
  - api.hyperliquid.xyz (candleSnapshot) pour l'historique du perp xyz:SHEIN
  - Yahoo Finance (0625.HK) via yfinance pour les clôtures journalières HKEX

Si l'accès à l'un des deux est différent chez toi (endpoint XYZ spécifique, autre broker
pour 0625.HK, export CSV manuel...), remplace juste les fonctions fetch_perp_candles()
et fetch_hkex_closes() par tes propres sources — le reste du pipeline ne change pas.

Avec seulement ~3 semaines d'historique coté (IPO le 1er sept. 2026), la calibration sera
bruitée. Le script te donne quand même les diagnostics nécessaires pour juger si le modèle
est exploitable ou s'il faut attendre plus de données / élargir la fenêtre d'estimation de tau.
------------------------------------------------------------------------------------------
"""

import json
import logging
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("shein_perp_signal.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("shein_perp_signal")

CALIBRATION_FILE = Path("calibration.json")

HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"
HKEX_TICKER = "0625.HK"
PERP_COIN = "xyz:SHEIN"  # nom exact du marché tel qu'exposé par l'API Hyperliquid (à vérifier)

# HKEX : 9:30-12:00 puis 13:00-16:00 heure de Hong Kong (UTC+8), fermé sam/dim.
HKEX_TZ_OFFSET_HOURS = 8
HKEX_CLOSE_HOUR_LOCAL = 16  # 16:00 HKT


# ----------------------------------------------------------------------------------------
# 1. RÉCUPÉRATION DES DONNÉES
# ----------------------------------------------------------------------------------------

def fetch_perp_candles(coin: str = PERP_COIN, interval: str = "15m", lookback_days: int = 60) -> pd.DataFrame:
    """Historique des bougies du perp via l'API publique Hyperliquid (candleSnapshot)."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    resp = requests.post(HYPERLIQUID_API, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw)
    # Colonnes typiques de l'API Hyperliquid : t (open time ms), T (close time ms), o,h,l,c,v
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"t": "timestamp", "c": "close"})
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)


def fetch_hkex_closes(ticker: str = HKEX_TICKER, lookback_days: int = 90) -> pd.DataFrame:
    """Clôtures journalières de 0625.HK via yfinance."""
    import yfinance as yf

    data = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)

    # yfinance >= 0.2.37 retourne parfois un MultiIndex de colonnes — on l'aplatit
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    # Clôture HKEX = 16:00 HKT ce jour-là, convertie en UTC
    data["close_time_utc"] = pd.to_datetime(data["date"]).dt.tz_localize(
        f"Etc/GMT-{HKEX_TZ_OFFSET_HOURS}"
    ).dt.tz_convert("UTC") + pd.Timedelta(hours=HKEX_CLOSE_HOUR_LOCAL - 0)
    return data[["close_time_utc", "close"]].rename(columns={"close": "hkex_close"})


# ----------------------------------------------------------------------------------------
# 2. CONSTRUCTION DU DATASET (r, tau, y) POUR CHAQUE OBSERVATION
# ----------------------------------------------------------------------------------------

def build_training_set(perp_df: pd.DataFrame, hkex_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque bougie perp comprise entre deux clôtures HKEX consécutives :
      - r   = rendement perp depuis la clôture HKEX précédente
      - tau = heures restantes avant la clôture HKEX suivante
      - y   = 1 si la clôture suivante est en hausse vs la précédente, sinon 0
    """
    rows = []
    hkex_df = hkex_df.sort_values("close_time_utc").reset_index(drop=True)

    for i in range(len(hkex_df) - 1):
        prev_close_time = hkex_df.loc[i, "close_time_utc"]
        prev_close_price = hkex_df.loc[i, "hkex_close"]
        next_close_time = hkex_df.loc[i + 1, "close_time_utc"]
        next_close_price = hkex_df.loc[i + 1, "hkex_close"]

        window = perp_df[
            (perp_df["timestamp"] > prev_close_time) & (perp_df["timestamp"] <= next_close_time)
        ]
        if window.empty:
            continue

        y = 1 if next_close_price > prev_close_price else 0

        for _, r_row in window.iterrows():
            r = r_row["close"] / prev_close_price - 1.0
            tau_hours = (next_close_time - r_row["timestamp"]).total_seconds() / 3600.0
            if tau_hours <= 0:
                continue
            rows.append({"r": r, "tau": tau_hours, "y": y})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# 3. CALIBRATION (régression probit)
# ----------------------------------------------------------------------------------------

def calibrate(train_df: pd.DataFrame):
    """
    Ajuste P(y=1) = Phi(gamma0 + gamma1 * r / sqrt(tau)) par probit.
    Retourne (gamma0, gamma1, résultat statsmodels complet pour diagnostics).
    """
    x = train_df["r"] / np.sqrt(train_df["tau"])
    X = sm.add_constant(x.rename("x"))
    y = train_df["y"]

    model = sm.Probit(y, X)
    result = model.fit(disp=0)

    gamma0, gamma1 = result.params["const"], result.params["x"]
    return gamma0, gamma1, result


# ----------------------------------------------------------------------------------------
# 4. PRÉDICTION EN TEMPS RÉEL
# ----------------------------------------------------------------------------------------

def predict_probability(r_now: float, tau_now_hours: float, gamma0: float, gamma1: float) -> float:
    """r_now : rendement perp depuis la dernière clôture HKEX (ex: 0.001 pour +0.1%)
       tau_now_hours : heures restantes avant la prochaine clôture HKEX"""
    z = gamma0 + gamma1 * r_now / np.sqrt(tau_now_hours)
    return float(norm.cdf(z))


# ----------------------------------------------------------------------------------------
# 5. EXEMPLE D'UTILISATION
# ----------------------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Démarrage de la calibration")
    try:
        log.info("Récupération des bougies perp Hyperliquid (60j)...")
        perp = fetch_perp_candles(lookback_days=60)
        log.info(f"{len(perp)} bougies perp récupérées")

        log.info("Récupération des clôtures HKEX (90j)...")
        hkex = fetch_hkex_closes(lookback_days=90)
        log.info(f"{len(hkex)} clôtures HKEX récupérées")

        train = build_training_set(perp, hkex)
        log.info(f"{len(train)} observations (r, tau, y) construites")

        if len(train) < 200:
            log.warning(
                "Peu d'observations (historique coté encore court) — "
                "coefficients indicatifs, pas fiables statistiquement."
            )

        gamma0, gamma1, result = calibrate(train)
        log.info(f"Calibration réussie : gamma0={gamma0:.4f}, gamma1={gamma1:.4f}")
        log.info("\n" + str(result.summary()))

        CALIBRATION_FILE.write_text(json.dumps({
            "gamma0": gamma0,
            "gamma1": gamma1,
            "n_observations": len(train),
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
        log.info(f"Calibration sauvegardée dans {CALIBRATION_FILE.resolve()}")

        # Exemples repris de la question : +0.1% de rendement perp, à 20h puis à 1h de la clôture
        for r_now, tau in [(0.001, 20), (0.001, 1)]:
            p = predict_probability(r_now, tau, gamma0, gamma1)
            log.info(f"Exemple : r={r_now:+.2%}, tau={tau}h  ->  P(hausse) = {p:.1%}")

    except Exception:
        log.exception("Échec de la calibration")
        raise
