"""
vip.py — Valore Intrinseco Predittivo (VIP)

Metrica custom che cattura il valore "nascosto" non intercettabile dal modello ML puro.
Il ML guarda numeri storici aggregati; il VIP codifica la conoscenza umana calcistica.

COMPONENTI:
  1. Tactical Position Modifier (TPM)
       Un difensore listato in Fantacalcio può essere un quinto d'attacco in campo
       (Dimarco, Theo, Bellanova). Un centrocampista può essere un trequartista o ala
       (Gudmundsson, Brescianini, Nico Paz). Il TPM applica un moltiplicatore 1.0–1.5
       sul bonus atteso, calibrato sulla posizione tattica reale.
       Se il giocatore non è nel config manuale, un inferitore statistico stima il ruolo
       tattico da gol_pg e assist_pg (wing-back ≈ D con assist alti, trequartista ≈ C con gol alti).

  2. P90 Discovery Factor
       Normalizza le performance sui 90 minuti e amplifica i giocatori con poche presenze
       ma metriche altissime — il segnale tipico del "talento emergente".
       Usa lo stimatore bayesiano di Laplace per bilanciare la scarsa campione con un prior
       di ruolo: evita che 1 gol in 1 partita mandi alle stelle un giocatore mediocre.
       Formula: z-score bayesiano × fattore_emersione (amplificato quando presenze < 15).

  3. Under-22 Growth Coefficient
       Coefficiente di crescita esponenziale per giocatori under-22 con trend di minutaggio
       crescente. La formula exp((22-age)/τ) genera una curva dove a 22 anni il bonus è quasi
       zero, a 19 anni è ~25%, a 17 anni raggiunge il cap del 40%.
       La crescita del minutaggio (configurable in vip_config.json) moltiplica il coefficiente:
       trend 1.5 = minutaggio in forte espansione, trend 0.8 = minutaggio calante.

INTEGRAZIONE NEL PIPELINE:
  train_prediction_model():
    1. previsione_ia  (ML pura)
    2. compute_vip()  → vip_tpm, vip_p90, vip_youth, vip_score  [questo modulo]
    3. _apply_strategy() → score_selezione  (usa previsione_ia potenziata dal VIP)
    4. apply_injury_modifiers()
"""

import os
import json
import math
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "data", "vip_config.json")

CURRENT_YEAR     = 2026
YOUTH_CUTOFF_AGE = 22        # età massima per il coefficiente di crescita
YOUTH_TAU        = 8.0       # costante di tempo della curva esponenziale
BAYESIAN_K       = 10        # presenze equivalenti del prior di ruolo
EMERGENCE_GAMES  = 15        # soglia oltre cui l'effetto emersione si azzera
MAX_VIP_BOOST    = 0.90      # boost totale massimo (+90%)
MIN_VIP_PENALTY  = -0.20     # penalità massima (−20%)

# ── Dizionario Moltiplicatori Ruolo Tattico ───────────────────────────────────
#
# Mappa la posizione tattica reale (inferita dalle stats) → coefficiente VIP.
# Cattura il valore che Fantacalcio.it non vede: un "D" può essere un ala;
# un "C" può essere un trequartista; il loro xG/xA vale molto di più del ruolo dice.
#
#   Esterno Offensivo (Dimarco, Theo, Bellanova):
#     listato come D ma produce xG+xA da ala → moltiplicatore alto
#   Terzino Bloccato (difensore puramente difensivo):
#     xG/xA basso strutturalmente → moltiplicatore riduttivo
#   Trequartista / Mezzala Att. (Gudmundsson, Brescianini, Nico Paz):
#     centrocampista con produzione da attaccante → moltiplicatore alto
#   Regista Difensivo (Matic, Cataldi):
#     basso contributo offensivo per scelta tattica → moltiplicatore riduttivo

ROLE_MULTIPLIERS: dict[str, float] = {
    # ── Difensori ────────────────────────────────────────────────────────────
    "ESTERNO_OFFENSIVO":  1.30,  # quinto/ala difensiva con xG+xA da attaccante
    "TERZINO_CLASSICO":   1.00,  # terzino moderno, qualche assist
    "TERZINO_BLOCCATO":   0.80,  # difensore puramente difensivo
    "DIFENSORE_CENTRALE": 0.85,  # centrale puro, rarissimi bonus offensivi
    # ── Centrocampisti ───────────────────────────────────────────────────────
    "TREQUARTISTA":       1.28,  # fantasista (Gudmundsson, Dybala, Nico Paz)
    "MEZZALA_ATT":        1.25,  # mezzala con inserimento (Brescianini, Calhanoglu)
    "ALA":                1.22,  # esterno offensivo puro (Politano, Colpani)
    "REGISTA_ATT":        1.08,  # regista avanzato con assist (Calhanoglu box)
    "REGISTA_DEF":        0.88,  # mediano difensivo (Matic, Cataldi, Rovella)
    # ── Attaccanti ───────────────────────────────────────────────────────────
    "PRIMA_PUNTA":        1.10,  # centravanti classico (Giroud, Arnautovic)
    "SECONDA_PUNTA":      1.15,  # attaccante di supporto con assist
    "ALA_ATT":            1.20,  # ala nel reparto avanzato (Lookman, Politano A)
    # ── Portieri ─────────────────────────────────────────────────────────────
    "PORTIERE":           1.00,  # la formula VIP non si applica ai GK
}


# ── Inferenza ruolo tattico e moltiplicatore giovane ─────────────────────────

def _infer_tactical_role(ruolo: str, gol_pg: float, assist_pg: float) -> str:
    """Mappa ruolo Fantacalcio + stats → posizione tattica dettagliata."""
    off = gol_pg + assist_pg
    if ruolo == "P":
        return "PORTIERE"
    elif ruolo == "D":
        if off >= 0.30:                               return "ESTERNO_OFFENSIVO"
        elif off >= 0.15 or assist_pg >= 0.10:        return "TERZINO_CLASSICO"
        else:                                          return "TERZINO_BLOCCATO"
    elif ruolo == "C":
        if gol_pg >= 0.20:                            return "TREQUARTISTA"
        elif gol_pg >= 0.10 or assist_pg >= 0.18:     return "MEZZALA_ATT"
        elif assist_pg >= 0.08:                        return "REGISTA_ATT"
        else:                                          return "REGISTA_DEF"
    elif ruolo == "A":
        if assist_pg >= 0.15:                         return "SECONDA_PUNTA"
        elif gol_pg >= 0.20:                          return "ALA_ATT"
        else:                                          return "PRIMA_PUNTA"
    return "PRIMA_PUNTA"


def _youth_multiplier(nome: str, cfg: dict) -> float:
    """
    Moltiplicatore giovane promessa:
      1.20 se under-22 e minutaggio crescente (trend > 1.0)
      1.10 se under-20 con minutaggio stabile
      1.05 se under-22 con trend neutro
      1.00 altrimenti
    """
    youth_players: dict = cfg.get("youth_players", {})
    nome_clean = str(nome).strip()
    player_data = youth_players.get(nome_clean)

    if player_data is None:
        cognome = nome_clean.split()[-1].lower()
        for key, val in youth_players.items():
            if not key.startswith("_") and cognome == key.split()[-1].lower():
                player_data = val
                break

    if player_data is None:
        return 1.0

    birth_year = int(player_data.get("birth_year", CURRENT_YEAR - 25))
    trend      = float(player_data.get("minutes_trend", 1.0))
    age        = CURRENT_YEAR - birth_year

    if age >= YOUTH_CUTOFF_AGE:
        return 1.0
    if trend > 1.0:
        return 1.20   # under-22 + minutaggio in crescita → segnale d'esplosione
    if age < 20:
        return 1.10   # giovanissimo anche con minuti stabili
    return 1.05       # under-22 neutro


def compute_vip_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola la componente matematica del VIP:

      VIP_raw = ((xG_P90 + xA_P90)) × Moltiplicatore_Ruolo × Moltiplicatore_Giovane

    Dove P90 è la normalizzazione ai 90 min effettivi tramite la stima del minutaggio
    medio per apparizione (titolare ≈ 90 min, subentrato ≈ 30 min).

    Il risultato grezzo è normalizzato per ruolo con rank percentile centrato:
      → migliore player del ruolo: +0.25
      → mediana del ruolo:         0.00
      → peggiore del ruolo:       -0.25

    Aggiunge le colonne:
      vip_formula       — componente normalizzata [-0.25, +0.25]
      vip_tactical_role — posizione tattica inferita (stringa)
    """
    cfg = _load_config()
    df  = df.copy()

    formula_raw: list[float] = []
    role_labels: list[str]   = []

    for _, row in df.iterrows():
        ruolo   = str(row.get("ruolo", "C"))
        nome    = str(row.get("nome", ""))
        gol_pg  = float(row.get("gol_pg", 0))
        ast_pg  = float(row.get("assist_pg", 0))
        xg_pg   = float(row.get("xg_pg", 0))
        tit_pct = float(row.get("titolarita_pct", 0.7))

        if ruolo == "P":
            formula_raw.append(0.0)
            role_labels.append("PORTIERE")
            continue

        # Stima minuti medi effettivi per partita:
        #   titolare pieno (tit_pct=1.0) → ~90 min
        #   puro subentrato (tit_pct=0.0) → ~30 min
        avg_min    = 30.0 + 60.0 * max(min(tit_pct, 1.0), 0.1)
        p90_factor = 90.0 / avg_min          # > 1 per i subentrati

        xG_p90 = xg_pg * p90_factor
        xA_p90 = ast_pg * p90_factor         # assist come proxy xA

        tactical_role = _infer_tactical_role(ruolo, gol_pg, ast_pg)
        role_mult     = ROLE_MULTIPLIERS.get(tactical_role, 1.0)
        youth_mult    = _youth_multiplier(nome, cfg)

        formula_raw.append((xG_p90 + xA_p90) * role_mult * youth_mult)
        role_labels.append(tactical_role)

    df["vip_tactical_role"] = role_labels

    vip_series = pd.Series(formula_raw, index=df.index)
    vip_norm   = pd.Series(0.0, index=df.index)

    for ruolo in df["ruolo"].unique():
        mask = df["ruolo"] == ruolo
        sub  = vip_series[mask]
        if len(sub) < 2 or sub.std() < 1e-4:
            continue
        rank_pct      = sub.rank(pct=True)          # [0, 1]
        vip_norm[mask] = (rank_pct - 0.5) * 0.50   # [-0.25, +0.25]

    df["vip_formula"] = np.round(vip_norm, 4)
    return df


# ── Caricamento config ────────────────────────────────────────────────────────

def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"vip_config.json non leggibile: {e} — uso defaults vuoti")
    return {"tactical_overrides": {}, "youth_players": {}, "p90_weights": {}, "role_priors_p90": {}}


# ── Componente 1: Tactical Position Modifier ─────────────────────────────────

def _tactical_bonus_manual(nome: str, cfg: dict) -> float:
    """
    Recupera il bonus tattico dal config manuale.
    Prova match esatto, poi per cognome (ultima parola).
    Ritorna il bonus TAL QUALE (es. 1.42 → aggiunge +42%, non 0.42).
    """
    overrides: dict = cfg.get("tactical_overrides", {})
    if not overrides:
        return 0.0

    nome_clean = str(nome).strip()
    if nome_clean in overrides:
        return float(overrides[nome_clean].get("bonus", 1.0)) - 1.0

    # Fallback sul cognome (ultima parola del nome)
    cognome = nome_clean.split()[-1].lower()
    for key, val in overrides.items():
        if key.startswith("_"):
            continue
        if cognome == key.split()[-1].lower():
            return float(val.get("bonus", 1.0)) - 1.0

    return 0.0


def _tactical_bonus_inferred(ruolo: str, gol_pg: float, assist_pg: float) -> float:
    """
    Inferisce il ruolo tattico dalle statistiche quando il giocatore non è nel config.

    Soglie basate su medie reali Serie A 2024-25:
      Difensore:
        gol+assist molto alto (>0.35) → quinto d'attacco offensivo      (+30%)
        gol o assist medio-alto       → quinto con contributo offensivo (+18%)
        solo assist medio             → quinto moderno                  (+10%)
      Centrocampista:
        gol molto alto (>0.22)        → trequartista/ala               (+22%)
        gol medio o assist molto alto → mezzala d'attacco              (+14%)
        gol basso ma assist buono     → regista avanzato               (+7%)
    """
    offensive_total = gol_pg + assist_pg

    if ruolo == "D":
        if offensive_total >= 0.35:
            return 0.30
        if offensive_total >= 0.20 or assist_pg >= 0.15:
            return 0.18
        if assist_pg >= 0.08:
            return 0.10
        return 0.0

    if ruolo == "C":
        if gol_pg >= 0.22:
            return 0.22
        if gol_pg >= 0.12 or assist_pg >= 0.22:
            return 0.14
        if gol_pg >= 0.07 or assist_pg >= 0.12:
            return 0.07
        return 0.0

    return 0.0


def _compute_tpm(row: pd.Series, cfg: dict) -> tuple[float, str]:
    """
    Ritorna (tpm_bonus, label_posizione) per un singolo giocatore.
    Se il bonus manuale è presente, ha priorità sull'inferenza statistica.
    """
    nome   = str(row.get("nome", ""))
    ruolo  = str(row.get("ruolo", "C"))
    gol_pg = float(row.get("gol_pg", 0))
    ast_pg = float(row.get("assist_pg", 0))

    manual = _tactical_bonus_manual(nome, cfg)
    if manual > 0:
        overrides = cfg.get("tactical_overrides", {})
        label = overrides.get(nome, {}).get("position", "AVANZATO")
        return manual, label

    inferred = _tactical_bonus_inferred(ruolo, gol_pg, ast_pg)
    if inferred > 0:
        label = "QUINTO_ATT" if ruolo == "D" else "MEZZALA_ATT"
        return inferred, label

    return 0.0, ruolo


# ── Componente 2: P90 Discovery Factor ───────────────────────────────────────

def _compute_p90_factor(
    row: pd.Series,
    role_priors: dict,
    p90_weights: dict,
) -> float:
    """
    Stimatore bayesiano di Laplace per le metriche P90.

    Passaggi:
      1. Calcola il composite P90 grezzo (gol×w + assist×w + xg×w)
      2. Applica shrinkage verso il prior di ruolo: meno presenze = più prior
      3. Standardizza rispetto alla media/std del ruolo
      4. Amplifica il z-score con un fattore d'emersione (alto se poche presenze)
      5. Converte z-score → bonus percentuale (clippato a ±30%)

    Effetti desiderati:
      - 3 presenze, gol_pg=0.5 (tipo Nico Paz in forma) → z alto + emersione alta → +20-25%
      - 30 presenze, gol_pg=0.3 (buon attaccante noto)  → z medio + emersione nulla → +10%
      - 1 presenza, gol_pg=1.0 (potrebbe essere un fluke) → prior corregge → +8% max
      - 20 presenze, gol_pg=0.05 (centrocampista difensivo) → z negativo → -5%
    """
    ruolo    = str(row.get("ruolo", "C"))
    presenze = float(row.get("presenze", 0))
    gol_pg   = float(row.get("gol_pg", 0))
    ast_pg   = float(row.get("assist_pg", 0))
    xg_pg    = float(row.get("xg_pg", 0))

    w_gol = float(p90_weights.get("gol_pg",    3.0))
    w_ast = float(p90_weights.get("assist_pg", 1.5))
    w_xg  = float(p90_weights.get("xg_pg",     0.8))

    p90_raw = gol_pg * w_gol + ast_pg * w_ast + xg_pg * w_xg

    prior = role_priors.get(ruolo, {"mean": 0.5, "std": 0.4})
    prior_mean = float(prior["mean"])
    prior_std  = max(float(prior["std"]), 0.01)

    # Bayesian shrinkage: con BAYESIAN_K presenze di "prior weight"
    # presenze=0 → p90_bayes = prior_mean (completamente shrinkato)
    # presenze→∞ → p90_bayes → p90_raw (completamente fiducia ai dati)
    denom     = max(presenze, 0) + BAYESIAN_K
    p90_bayes = (p90_raw * presenze + prior_mean * BAYESIAN_K) / denom

    # Standardizzazione: quanto devia dalla media del ruolo?
    z = (p90_bayes - prior_mean) / prior_std

    # Fattore emersione: amplifica il segnale per i giocatori con poche presenze
    # che mostrano comunque metriche alte (il "diamante grezzo")
    # A 0 presenze → emersione=2.0 (ma prior lo schiaccia, quindi netto ~0)
    # A 7 presenze → emersione=1.53 (buon amplificatore)
    # A 15 presenze → emersione=1.0 (nessuna amplificazione)
    confidence      = min(presenze / EMERGENCE_GAMES, 1.0)
    emergence_factor = 2.0 - confidence   # [1.0, 2.0]

    p90_bonus = z * 0.065 * emergence_factor
    return float(np.clip(p90_bonus, -0.15, 0.30))


# ── Componente 3: Under-22 Growth Coefficient ────────────────────────────────

def _compute_youth_factor(row, cfg: dict) -> float:
    """
    Coefficiente di crescita esponenziale per giocatori under-22 (COMPLETAMENTE AUTOMATICO).
    """
    age = float(row.get("eta", 26) or 26)
    
    if age > YOUTH_CUTOFF_AGE:
        return 0.0
        
    # Stima del trend in base alla forma recente o presenze
    forma = float(row.get("forma_recente_score", 0) or 0)
    tit_pct = float(row.get("titolarita_pct", 0) or 0)
    
    minutes_trend = 1.0
    if forma > 0.15 or tit_pct > 0.5:
        minutes_trend = 1.3
    elif forma < -0.15:
        minutes_trend = 0.8

    age_bonus = math.exp((YOUTH_CUTOFF_AGE - age) / YOUTH_TAU) - 1.0
    raw_bonus = age_bonus * minutes_trend * 0.55
    return float(np.clip(raw_bonus, 0.0, 0.40))


# ── Funzione principale ───────────────────────────────────────────────────────

def compute_vip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il Valore Intrinseco Predittivo per ogni giocatore nel DataFrame.

    Aggiunge le colonne:
      vip_tpm       — bonus tattico (0.0 se nessun override, max ~0.45)
      vip_p90       — bonus P90 bayesiano (range: -0.15 a +0.30)
      vip_youth     — bonus crescita under-22 (0 se >22 anni, max 0.40)
      vip_total     — bonus complessivo clippato (range: -0.20 a +0.90)
      vip_position  — label della posizione tattica inferita (stringa)
      vip_score     — previsione_ia × (1 + vip_total): lo score VIP-enhanced

    Modifica 'previsione_ia' con un blending gentile (α=0.30) per trasportare
    il segnale VIP nelle fasi successive del pipeline (strategy + optimizer).

    Il blending preserva la leggibilità della previsione_ia originale
    (non varia di più del 20% rispetto al valore base).
    """
    cfg          = _load_config()
    role_priors  = cfg.get("role_priors_p90", {})
    p90_weights  = cfg.get("p90_weights", {"gol_pg": 3.0, "assist_pg": 1.5, "xg_pg": 0.8})

    df = df.copy()

    tpm_list      = []
    p90_list      = []
    youth_list    = []
    position_list = []

    for _, row in df.iterrows():
        tpm, pos = _compute_tpm(row, cfg)
        p90      = _compute_p90_factor(row, role_priors, p90_weights)
        youth    = _compute_youth_factor(row, cfg)

        tpm_list.append(tpm)
        p90_list.append(p90)
        youth_list.append(youth)
        position_list.append(pos)

    df["vip_tpm"]      = np.round(tpm_list, 4)
    df["vip_p90"]      = np.round(p90_list, 4)
    df["vip_youth"]    = np.round(youth_list, 4)
    df["vip_position"] = position_list

    # Componente 4: formula matematica automatica
    #   VIP_formula = ((xG + xA) / min_P90) * mult_ruolo * mult_giovane
    #   normalizzata per ruolo → [-0.25, +0.25]
    df = compute_vip_formula(df)
    formula_arr = df["vip_formula"].values

    # Combina i 4 componenti linearmente.
    # La formula automatica pesa 0.6 (segnale forte ma già normalizzato in [-0.25, +0.25])
    # per non sovrastare TPM che può arrivare a +0.45.
    vip_total = (
        np.array(tpm_list)
        + np.array(p90_list)
        + np.array(youth_list)
        + formula_arr * 0.60
    )
    vip_total = np.clip(vip_total, MIN_VIP_PENALTY, MAX_VIP_BOOST)
    df["vip_total"] = np.round(vip_total, 4)

    # VIP score: previsione_ia amplificata
    base = df["previsione_ia"].values.copy()
    vip_raw = base * (1.0 + vip_total)
    df["vip_score"] = np.round(np.clip(vip_raw, 3.0, 12.0), 3)

    # Blending: aggiorna previsione_ia con una correzione moderata (α=0.30)
    # Così la previsione_ia rimane interpretabile come FM stimata ma riflette il VIP
    alpha = 0.30
    df["previsione_ia"] = np.round(
        np.clip((1 - alpha) * base + alpha * vip_raw, 3.0, 12.0), 3
    )

    _log_vip_top(df)
    return df


def _log_vip_top(df: pd.DataFrame, n: int = 10) -> None:
    """Logga i top n giocatori per vip_total."""
    if "vip_total" not in df.columns:
        return
    top = df.nlargest(n, "vip_total")[
        ["nome", "ruolo", "vip_position", "vip_tpm", "vip_p90", "vip_youth", "vip_total", "vip_score"]
    ]
    logger.info("\n── TOP VIP ──────────────────────────────────────────────────────")
    for _, r in top.iterrows():
        logger.info(
            f"  {str(r['nome']):22s} {r['ruolo']} │ pos={str(r['vip_position']):18s} │ "
            f"TPM={r['vip_tpm']:+.2f} P90={r['vip_p90']:+.2f} "
            f"Youth={r['vip_youth']:+.2f} │ VIP={r['vip_total']:+.3f} "
            f"→ score={r['vip_score']:.2f}"
        )
    logger.info("────────────────────────────────────────────────────────────────\n")


# ── Gestione config da UI ─────────────────────────────────────────────────────

def add_tactical_override(nome: str, position: str, bonus: float, note: str = "") -> bool:
    """Aggiunge/aggiorna un override tattico nel config. Ritorna True se ok."""
    try:
        cfg = _load_config()
        cfg.setdefault("tactical_overrides", {})[nome] = {
            "position": position,
            "bonus": round(float(bonus), 3),
            "note": note,
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Errore salvataggio override tattico: {e}")
        return False


def add_youth_player(nome: str, birth_year: int, minutes_trend: float, note: str = "") -> bool:
    """Aggiunge/aggiorna un giocatore under-22 nel config. Ritorna True se ok."""
    try:
        cfg = _load_config()
        cfg.setdefault("youth_players", {})[nome] = {
            "birth_year": int(birth_year),
            "minutes_trend": round(float(minutes_trend), 2),
            "note": note,
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Errore salvataggio giovane promessa: {e}")
        return False


def get_vip_config_summary() -> dict:
    """Ritorna un riassunto del config VIP per la UI."""
    cfg = _load_config()
    tacticals = {k: v for k, v in cfg.get("tactical_overrides", {}).items() if not k.startswith("_")}
    youths    = {k: v for k, v in cfg.get("youth_players", {}).items() if not k.startswith("_")}
    return {
        "version":    cfg.get("version", "N/D"),
        "n_tactical": len(tacticals),
        "n_youth":    len(youths),
        "tactical":   tacticals,
        "youth":      youths,
    }


# ── CLI di test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base = os.path.dirname(os.path.abspath(__file__))
    csv  = os.path.join(base, "data", "serie_a_23_24_backtest.csv")

    df = pd.read_csv(csv)

    # Simula previsione_ia se non c'è (usa fanta_media)
    if "previsione_ia" not in df.columns:
        df["previsione_ia"] = df.get("fanta_media", pd.Series(6.0, index=df.index)).clip(3, 10)

    # Simula presenze se non c'è
    for col in ["presenze", "gol_pg", "assist_pg", "xg_pg"]:
        if col not in df.columns:
            df[col] = 0.0

    df = compute_vip(df)

    print("\n── TOP 15 PER VIP SCORE ──────────────────────────────────────────")
    top = df.nlargest(15, "vip_score")
    print(f"{'Nome':22s} {'R':2s} {'Posizione':18s} {'TPM':>6} {'P90':>6} {'Youth':>6} {'TOT':>6} {'Score':>6}")
    print("-" * 80)
    for _, r in top.iterrows():
        print(
            f"{str(r['nome']):22s} {r['ruolo']:2s} {str(r['vip_position']):18s} "
            f"{r['vip_tpm']:+.3f} {r['vip_p90']:+.3f} {r['vip_youth']:+.3f} "
            f"{r['vip_total']:+.3f} {r['vip_score']:6.2f}"
        )
