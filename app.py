import streamlit as st
import pandas as pd
import importlib
import os
import time
import glob

import importer as _imp
import optimizer as _opt
import predict as _pred
import predict_market as _pm
import simulator as _sim
import scraper_fantacalcio as _scr
import scraper_transfermarkt as _tm
import scraper_injuries as _inj
importlib.reload(_imp)
importlib.reload(_opt)
importlib.reload(_pred)
importlib.reload(_pm)
importlib.reload(_sim)
importlib.reload(_scr)
importlib.reload(_tm)
importlib.reload(_inj)
import auction_copilot as _auc
importlib.reload(_auc)

from importer import import_real_quotazioni
from optimizer import optimize_team
from predict import train_prediction_model

@st.cache_data(show_spinner=False)
def _get_cached_predictions(df, strat):
    # Usiamo _df solo per invalidare la cache se cambia
    if df is not None:
        df.to_csv(INPUT_TEMP, index=False)
        return train_prediction_model(INPUT_TEMP, strategy=strat)
    return None
from simulator import run_performance_comparison
from scraper_fantacalcio import fetch_and_save_stats, save_manual_excel, load_cached_stats
from scraper_transfermarkt import fetch_and_save_transfers, load_cached_transfers
from scraper_injuries import fetch_and_save_injuries, load_cached_injuries, get_injury_summary

from auction_copilot import AuctionState, calculate_copilot_bids
from foreign_league import enrich_dataset_with_foreign_arrivals
from set_pieces import apply_set_pieces_boost
from predict_market import apply_real_market_logic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_TEMP   = os.path.join(DATA_DIR, "input_temp.csv")
DEFAULT_DATA = os.path.join(DATA_DIR, "serie_a_23_24_backtest.csv")
INJURIES_CACHE_PATH  = os.path.join(DATA_DIR, "injuries_cache.json")

INJURIES_STALE_HOURS = 12   # auto-refresh infortuni ogni 12 ore

# ── Helper: età cache ─────────────────────────────────────────────────────────
def _cache_is_stale(path: str, max_age_hours: float) -> bool:
    if not os.path.exists(path):
        return True
    return (time.time() - os.path.getmtime(path)) / 3600 > max_age_hours

def _cache_age_str(path: str) -> str:
    if not os.path.exists(path):
        return "mai aggiornata"
    age_h = (time.time() - os.path.getmtime(path)) / 3600
    if age_h < 1:
        return f"{int(age_h*60)} min fa"
    if age_h < 24:
        return f"{age_h:.0f}h fa"
    return f"{age_h/24:.0f}g fa"

# ── Helper: mappa nome strategia → chiave interna ─────────────────────────────
def _strategy_key(label: str) -> str:
    if "Aggressiva" in label: return "aggressiva"
    if "Moneyball" in label: return "moneyball"
    if "Sprint" in label: return "sprint_calendario"
    return "master"


# ── Spiegazione IA ────────────────────────────────────────────────────────────
_RUOLO_LABEL = {"P": "Portiere", "D": "Difensore", "C": "Centrocampista", "A": "Attaccante"}

_STRATEGY_INTRO = {
    "listone": (
        "La strategia **Listone Master** è ideata per vincere i campionati a listone: "
        "l'IA iper-concentra l'85-92% del budget sui **11 Super-Top Titolari** (selezionando il modulo ideale "
        "es. 3-4-3 o 4-3-3 col modificatore difesa) e completa la rosa con 14 coperture a 1 credito "
        "a titolarità garantita per non bucare mai la formazione."
    ),
    "conservativa": (
        "La strategia **Conservativa** privilegia la continuità: l'IA bilancia "
        "la fanta_media storica (65%) con il punteggio del modello (35%), "
        "scegliendo giocatori affidabili che rendono con costanza settimana dopo settimana."
    ),
    "aggressiva": (
        "La strategia **Aggressiva** punta all'esplosività: l'IA aggiunge un bonus "
        "diretto proporzionale a gol/partita, assist/partita e xG, "
        "privilegiando chi ha il maggior potenziale offensivo anche a fronte di qualche "
        "giornata in bianco."
    ),
    "scommesse": (
        "La strategia **Scommesse** cerca valore nascosto: l'IA moltiplica il punteggio "
        "inversamente al costo, premiando i giocatori sottovalutati dal mercato. "
        "Include un bonus extra per chi ha appena cambiato squadra in meglio."
    ),
}


def _get_inj(nome: str) -> tuple[str, float]:
    """Ritorna (descrizione_evento, modifier) dal cache infortuni, o ('', 1.0) se assente."""
    try:
        cache = load_cached_injuries()
        for k, v in cache.items():
            if nome.lower() in k.lower() or k.lower() in nome.lower():
                return v.get("tipo", ""), float(v.get("modifier", 1.0))
    except Exception:
        pass
    return "", 1.0


_GOL_BONUS_RUOLO = {"P": 0.0, "D": 4.0, "C": 3.0, "A": 3.0}

def _explain_player(row: pd.Series, strategy: str, rank: int) -> str:
    """Genera 2-4 righe di spiegazione per un singolo giocatore."""
    nome = row.get("nome", "?")
    ruolo = row.get("ruolo", "?")
    prev = float(row.get("previsione_ia", 0))
    score_sel = row.get("score_selezione", None)
    fm = row.get("fanta_media", None)
    costo = int(row.get("costo_iniziale", 0))
    gol = float(row.get("gol_pg", 0))
    ast = float(row.get("assist_pg", 0))
    xg = float(row.get("xg_pg", 0))
    amm = float(row.get("ammonizioni_pg", 0))
    tit = float(row.get("titolarita_pct", 0))
    cambio = int(row.get("cambio_squadra", 0))
    upgrade = int(row.get("upgrade_squadra", 0))
    tipo_p = int(row.get("tipo_prestito", 0))
    inj_tipo, inj_mod = _get_inj(nome)

    ruolo_str = _RUOLO_LABEL.get(ruolo, ruolo)
    lines: list[str] = []

    # Riga principale: FM prevista (realistica) + priorità strategia
    fm_str = f" | FM storica: {float(fm):.2f}" if fm and float(fm) > 4.5 else ""
    sel_str = f" | Priorità strategia: {float(score_sel):.2f}" if score_sel is not None else ""
    lines.append(
        f"**{nome}** ({ruolo_str}, {costo} cr.) — FM prevista: **{prev:.2f}**{fm_str}{sel_str}"
    )

    # Motivazione statistica per ruolo
    stat_parts: list[str] = []
    if ruolo == "P":
        pct = int(tit * 100)
        stat_parts.append(f"titolare nel {pct}% delle partite")
        if amm < 0.05:
            stat_parts.append("pochissime ammonizioni")
    else:
        if gol >= 0.25:
            stat_parts.append(f"{gol:.2f} gol/partita")
        if ast >= 0.15:
            stat_parts.append(f"{ast:.2f} assist/partita")
        if xg >= 0.15:
            stat_parts.append(f"xG {xg:.2f}/partita")
        if tit >= 0.80:
            stat_parts.append(f"titolare fisso ({int(tit*100)}%)")
        elif tit >= 0.60:
            stat_parts.append(f"buona titolarità ({int(tit*100)}%)")
        if amm >= 0.25:
            stat_parts.append(f"⚠️ {amm:.2f} ammonizioni/g (penalità)")

    if stat_parts:
        lines.append("  → " + ", ".join(stat_parts))

    # Effetti trasferimento
    if cambio == 1:
        if upgrade == 1:
            lines.append(
                "  → **Trasferimento positivo**: passato a una squadra di livello superiore, "
                "più possibilità di titolarità e bonus in un contesto offensivo migliore."
            )
        else:
            lines.append(
                "  → Ha cambiato squadra quest'estate: nuovo ambiente, motivazione alta, "
                "da monitorare nelle prime giornate."
            )
    if tipo_p == 1:
        lines.append(
            "  → In prestito: minuti non garantiti a lungo termine — l'IA ha scontato "
            "leggermente il suo score per questa instabilità contrattuale."
        )

    # Logica strategia — spiega perché il giocatore ha una priorità alta/bassa
    s = strategy.lower()
    if "conserv" in s and fm and float(fm) > 5.5:
        lines.append(
            f"  → **Perché selezionato (Conservativa)**: FM storica {float(fm):.2f} pesa 75% "
            f"nello score di selezione ({float(score_sel):.2f}) — rendimento già dimostrato."
        )
    elif "agg" in s and (gol + ast) > 0.3 and ruolo != "P":
        off = round(gol * _GOL_BONUS_RUOLO.get(ruolo, 3.0) + ast + xg * 0.5, 2)
        lines.append(
            f"  → **Perché selezionato (Aggressiva)**: output offensivo {off:.2f} "
            f"(gol×{_GOL_BONUS_RUOLO.get(ruolo,3.0):.0f} + assist + xG×0.5) "
            f"→ score selezione {float(score_sel):.2f}."
        )
    elif ("sco" in s or "hype" in s) and score_sel is not None:
        ratio = round(prev / max(costo, 1), 3)
        lines.append(
            f"  → **Perché selezionato (Scommesse)**: FM prevista {prev:.2f} / "
            f"{costo} cr. = rapporto {ratio} — "
            f"score selezione {float(score_sel):.2f} (la FM prevista reale resta {prev:.2f})."
        )

    # Infortuni
    if inj_mod < 1.0:
        sconto = int((1 - inj_mod) * 100)
        lines.append(
            f"  → ⚠️ **{inj_tipo}** rilevato — score ridotto del {sconto}% dal sistema infortuni. "
            "Valuta se inserirlo comunque o cercare un'alternativa."
        )

    return "\n".join(lines)


def _explain_team(titolari: pd.DataFrame, team: pd.DataFrame, strategy: str,
                  strategy_label: str, budget: int) -> str:
    """Genera la spiegazione completa della rosa scelta dall'IA."""
    costo_tot = int(team["costo_iniziale"].sum())
    budget_rimasto = budget - costo_tot
    n_riserve = len(team) - len(titolari)
    prev_tot = round(titolari["previsione_ia"].sum(), 2)

    # Intro strategia
    intro = _STRATEGY_INTRO.get(strategy, "")

    # Sommario rosa
    role_counts = titolari["ruolo"].value_counts().to_dict()
    modulo = f"{role_counts.get('D',0)}-{role_counts.get('C',0)}-{role_counts.get('A',0)}"
    sommario = (
        f"**Rosa selezionata**: modulo {modulo} | "
        f"Budget usato: **{costo_tot}/{budget} cr.** (avanza {budget_rimasto} cr.) | "
        f"{n_riserve} riserve | Punteggio atteso titolari: **{prev_tot}**"
    )

    # Spiegazione per giocatore (solo titolari, ordinati per ruolo)
    ordine_ruoli = ["P", "D", "C", "A"]
    player_lines: list[str] = []
    for ruolo in ordine_ruoli:
        gruppo = titolari[titolari["ruolo"] == ruolo]
        if gruppo.empty:
            continue
        player_lines.append(f"\n**{'Portieri' if ruolo=='P' else 'Difensori' if ruolo=='D' else 'Centrocampisti' if ruolo=='C' else 'Attaccanti'}**")
        for i, (_, row) in enumerate(gruppo.iterrows(), 1):
            player_lines.append(_explain_player(row, strategy, i))

    return intro + "\n\n" + sommario + "\n" + "\n\n".join(player_lines)

# ── Layout & UI Theme ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FantaBot AI 2026/27 — Market Master",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Dark Theme Background & Cards ── */
.stApp {
    background-color: #0b0e14 !important;
}

/* ── Hero Banner Header ── */
.hero-header {
    background: linear-gradient(135deg, rgba(24, 28, 43, 0.95) 0%, rgba(15, 20, 32, 0.98) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #00E676 0%, #00B0FF 50%, #7C4DFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px !important;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    color: #94A3B8 !important;
    margin-bottom: 16px !important;
}

.hero-badge-container {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

.hero-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.9rem !important;
    font-weight: 600;
    color: #E2E8F0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.hero-badge-green {
    border-color: rgba(0, 230, 118, 0.4);
    color: #00E676;
    background: rgba(0, 230, 118, 0.08);
}

.hero-badge-purple {
    border-color: rgba(124, 77, 255, 0.4);
    color: #B388FF;
    background: rgba(124, 77, 255, 0.08);
}

.hero-badge-gold {
    border-color: rgba(255, 215, 0, 0.4);
    color: #FFD700;
    background: rgba(255, 215, 0, 0.08);
}

/* ── Sidebar Custom Styling & Contrast ── */
[data-testid="stSidebar"] {
    background-color: #11151F !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #00E676 !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] small, 
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] div.caption {
    color: #CBD5E1 !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"], 
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section, 
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div,
[data-baseweb="file-uploader"],
[data-baseweb="file-uploader"] > div {
    background-color: #151923 !important;
    color: #F8FAFC !important;
}

[data-testid="stFileUploader"] section,
[data-baseweb="file-uploader"] {
    border: 1px dashed rgba(0, 230, 118, 0.4) !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] section:hover,
[data-baseweb="file-uploader"]:hover {
    background-color: rgba(0, 230, 118, 0.08) !important;
    border-color: rgba(0, 230, 118, 0.8) !important;
}

[data-testid="stFileUploadDropzone"] * ,
[data-testid="stFileUploader"] * {
    color: #E2E8F0 !important;
}
/* ── File Uploader Button Overlay ── */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%) !important;
    color: #0B0E14 !important;
    border: none !important;
    font-weight: 700 !important;
}

/* ── Metric Box Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01)) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #00E676 !important;
}

/* ── Tab Header Styling ── */
[data-baseweb="tab-list"] {
    gap: 8px !important;
    background-color: rgba(255, 255, 255, 0.02) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
}

[data-baseweb="tab"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 8px 18px !important;
    color: #94A3B8 !important;
    border: none !important;
    transition: all 0.2s ease-in-out !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%) !important;
    color: #0B0E14 !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 12px rgba(0, 230, 118, 0.3) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00E676 0%, #00B0FF 100%) !important;
    color: #0B0E14 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-size: 1.05rem !important;
    box-shadow: 0 4px 14px rgba(0, 230, 118, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 230, 118, 0.4) !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    overflow: hidden !important;
}

/* ── Streamlit Alert Banners (st.info, st.warning, st.success, st.error) ── */
[data-testid="stAlert"], div[data-baseweb="notification"] {
    background-color: rgba(18, 22, 33, 0.95) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
}

[data-testid="stAlert"] * {
    color: #E2E8F0 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"], div.streamlit-expanderContent, details {
    background-color: rgba(18, 22, 33, 0.9) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
}

[data-testid="stExpander"] details summary, details summary p {
    color: #00E676 !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}

/* ── Form Inputs, Text Areas, Number Inputs, Selectboxes ── */
input[type="text"], input[type="number"], input[type="password"], textarea,
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] > div {
    background-color: #181C2B !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
}

ul[role="listbox"], [data-baseweb="menu"] {
    background-color: #181C2B !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #F8FAFC !important;
}

li[role="option"] {
    background-color: #181C2B !important;
    color: #E2E8F0 !important;
}

li[role="option"]:hover {
    background-color: #262C3E !important;
    color: #00E676 !important;
}

/* ── Radio & Checkbox Widgets ── */
[data-testid="stMarkdownContainer"] p {
    color: #E2E8F0 !important;
}

[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(255, 255, 255, 0.1) !important;
}

/* ── Top Header Bar ── */
header[data-testid="stHeader"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

import base64

avatar_path = os.path.join(BASE_DIR, "francy_avatar.jpeg")
if os.path.exists(avatar_path):
    with open(avatar_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    avatar_html = f'<img src="data:image/jpeg;base64,{encoded}" style="width: 90px; height: 90px; border-radius: 50%; vertical-align: middle; margin-bottom: 8px; margin-right: 15px; border: 2px solid #5a32fa; box-shadow: 0 0 10px rgba(90, 50, 250, 0.5); object-fit: cover;">'
else:
    avatar_html = "🤖"

st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">{avatar_html} FantaBot AI 2026/27 — Market Master</div>
    <div class="hero-subtitle">Agente AI per Fantacalcio: Previsioni Avanzate, Modello Tattico, Rigoristi e Copilota d'Asta Live in Tempo Reale.</div>
    <div class="hero-badge-container">
        <span class="hero-badge hero-badge-green">● AI Engine 2026/27 Attivo</span>
        <span class="hero-badge hero-badge-purple">⚡ Live Auction Copilot Ready</span>
        <span class="hero-badge hero-badge-gold">⚽ Rigoristi & Tactical VIP Integration</span>
        <span class="hero-badge">🔄 Auto-News Scraper (30m)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Parametri Asta")
num_partecipanti = st.sidebar.selectbox(
    "Partecipanti (Modifica Tetto Asta)",
    [8, 10, 12],
    index=1,
    help="Modifica dinamicamente le percentuali del budget (es. a 10 e 12 riduce budget attaccanti e alza la difesa per Modificatore)."
)
budget = st.sidebar.number_input(
    "Budget crediti (digita o usa ↑↓)",
    min_value=100, max_value=1000, value=500, step=1,
    key="budget_input",
    help="Digita il valore esatto o usa le frecce. Range: 100–1000 cr.",
)
strategy_label = st.sidebar.selectbox(
    "🎯 Strategia Asta",
    ["🏆 Listone Master (Equilibrata)", "⚔️ Aggressiva (Trazione Anteriore)", "💎 Moneyball (VIP & Sottovalutati)", "🚀 Sprint Iniziale (Focus Calendario)"],
    help=(
        "Master: cerca i migliori 11 titolari e riserve a 1 cr equilibrando il budget.\n"
        "Aggressiva: budget quasi tutto su centrocampo e attacco, difesa low cost.\n"
        "Moneyball: cerca giocatori con statistiche tattiche nascoste ad alto potenziale.\n"
        "Sprint Iniziale: massimizza i giocatori con calendario super-facile."
    ),
)
strategy = _strategy_key(strategy_label)

# ── 1. DATI DEL NUOVO CAMPIONATO ──────────────────────────────────────────────────────
st.sidebar.subheader("1️⃣ Listone e Forma Attuale")
_fc_cache_path = os.path.join(DATA_DIR, "fantacalcio_stats_cache.csv")

with st.sidebar.expander("📥 LISTONE E VOTI REALI", expanded=True):
    st.caption("**1. Carica il file Listone Quotazioni (Excel/CSV)**")
    fc_excel_file = st.file_uploader(
        "File Quotazioni", type=["xlsx", "xls", "csv"], key="fc_excel_upload"
    )
    
    st.caption("**2. Incolla i Voti (Prime giornate)** se il campionato è iniziato")
    fc_stats_text = st.text_area("Copia-incolla la tabella voti", height=100)
    
    if st.button("💾 Unisci e Salva"):
        if fc_excel_file is not None:
            with st.sidebar.status("Importazione in corso...") as s:
                try:
                    from importer import import_real_quotazioni
                    import tempfile
                    import os
                    import pandas as pd
                    import numpy as np
                    
                    dfs = []
                    
                    # 1. Excel (Quotazioni)
                    ext = os.path.splitext(fc_excel_file.name)[1]
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f_tmp:
                        f_tmp.write(fc_excel_file.getvalue())
                        tmp_name = f_tmp.name
                    df_q = import_real_quotazioni(tmp_name)
                    if df_q is not None and not df_q.empty:
                        dfs.append(df_q)
                        
                    # 2. Testo (Statistiche / Voti)
                    if fc_stats_text.strip():
                        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f_tmp:
                            f_tmp.write(fc_stats_text)
                            tmp_name2 = f_tmp.name
                        df_s = import_real_quotazioni(tmp_name2)
                        if df_s is not None and not df_s.empty:
                            dfs.append(df_s)
                            
                    if dfs:
                        dfs_indexed = [d.set_index('nome') for d in dfs]
                        final_df = dfs_indexed[0]
                        for other_df in dfs_indexed[1:]:
                            if 'costo_iniziale' in final_df:
                                final_df['costo_iniziale'] = final_df['costo_iniziale'].replace(1, np.nan)
                            if 'costo_iniziale' in other_df:
                                other_df['costo_iniziale'] = other_df['costo_iniziale'].replace(1, np.nan)
                            final_df = final_df.combine_first(other_df)
                        
                        if 'costo_iniziale' in final_df:
                            final_df['costo_iniziale'] = final_df['costo_iniziale'].fillna(1)
                            
                        new_df = final_df.reset_index()
                        QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
                        new_df.to_csv(QUOTAZIONI_PATH, index=False)
                        st.session_state["last_uploaded_file"] = fc_excel_file.name
                        _get_cached_predictions.clear()
                        s.update(label=f"✅ {len(new_df)} giocatori uniti con successo!", state="complete", expanded=False)
                    else:
                        s.update(label="❌ Nessun dato valido", state="error", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ Errore: {e}", state="error")
        else:
            st.error("Devi prima caricare il file Excel delle Quotazioni!")

# ── 2. DATI STORICI FBREF (IL CERVELLO IA) ─────────────────────────────────────────
st.sidebar.subheader("2️⃣ Dati Storici (Motore IA)")
with st.sidebar.expander("📈 Incolla Statistiche FBRef", expanded=True):
    st.caption(
        "**Recupera le statistiche tattiche della SCORSA stagione (es. 25/26).**\n\n"
        "*(FBRef blocca l'accesso automatico con Cloudflare, devi copiarle tu)*\n\n"
        "1. Apri [FBRef Serie A (Stagione Scorsa)](https://fbref.com/it/comps/11/history/Serie-A-Seasons)\n"
        "2. Scorri fino alla tabella 'Standard Stats'\n"
        "3. Clicca 'Share & Export' → 'Get table as CSV'\n"
        "4. Incolla il testo qui sotto:"
    )
    fbref_csv_text = st.text_area("Incolla CSV FBref", height=100)
    if st.button("💾 Salva Dati Storici"):
        if fbref_csv_text.strip():
            try:
                import pandas as pd
                import io
                from scraper_stats import _normalize_fbref_df
                
                # Il CSV esportato da FBRef ha una doppia riga di intestazione.
                # Se "Player" è nella seconda riga, usiamo header=1.
                lines = fbref_csv_text.strip().split('\n')
                if len(lines) > 1 and "Player" in lines[1] and "Player" not in lines[0]:
                    df_raw = pd.read_csv(io.StringIO(fbref_csv_text), header=1)
                else:
                    df_raw = pd.read_csv(io.StringIO(fbref_csv_text))
                    
                df_norm = _normalize_fbref_df(df_raw)
                if df_norm is not None and not df_norm.empty:
                    df_norm.to_csv(os.path.join(DATA_DIR, "fbref_stats_cache.csv"), index=False)
                    _get_cached_predictions.clear()
                    st.success(f"✅ Dati Storici tradotti e salvati ({len(df_norm)} giocatori)!")
                else:
                    st.error("❌ Errore nella traduzione del CSV. Assicurati di aver copiato la tabella 'Standard Stats'.")
            except Exception as e:
                st.error(f"❌ Errore di formattazione: {e}")
        else:
            st.error("Incolla il testo prima di salvare.")

# ── Mercato ───────────────────────────────────────────────────────────────────
st.sidebar.subheader("🔀 Mercato")
with st.sidebar.expander("Aggiorna Trasferimenti"):
    st.caption("Formato manuale: Nome;DaSquadra;ASquadra;tipo (una riga per trasferimento)")
    manual_transfers = st.text_area(
        "Trasferimenti manuali (opzionale)",
        placeholder="Vlahovic;Juventus;Arsenal;definitivo",
        height=80, key="manual_transfers",
    )
    season_year = st.number_input("Anno stagione", min_value=2020, max_value=2030, value=2026, step=1)
    if st.button("⬇️ Salva Trasferimenti"):
        with st.spinner("Elaborazione..."):
            ok, msg = fetch_and_save_transfers(int(season_year), manual_transfers)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

cached_tm = load_cached_transfers()
tm_cache_path = os.path.join(DATA_DIR, "transfers_cache.csv")

TRANSFERS_STALE_HOURS = 12
if _cache_is_stale(tm_cache_path, TRANSFERS_STALE_HOURS):
    with st.sidebar.status("🔄 Aggiornamento automatico trasferimenti...") as s:
        try:
            ok, msg_txt = fetch_and_save_transfers(int(season_year))
            s.update(label=f"✅ Trasferimenti aggiornati" if ok else f"⚠️ {msg_txt}", state="complete" if ok else "error")
            cached_tm = load_cached_transfers()
        except Exception as e:
            s.update(label=f"❌ Errore trasferimenti: {e}", state="error")

st.sidebar.caption(
    f"Trasferimenti: {len(cached_tm)} in cache — {_cache_age_str(tm_cache_path)}"
    if cached_tm is not None else "Trasferimenti: nessuna cache"
)

# ── News calciomercato (auto-caricamento) ─────────────────────────────────────
st.sidebar.subheader("📰 News Calciomercato")

_NEWS_STALE_MINUTES = 30

def _news_is_stale() -> bool:
    ts = st.session_state.get("news_fetched_at", 0)
    return (time.time() - ts) > (_NEWS_STALE_MINUTES * 60)

AUTO_NEWS_PATH = os.path.join(DATA_DIR, "automated_news.txt")

if "news_area" not in st.session_state or not st.session_state["news_area"] or _news_is_stale():
    try:
        from news_scraper import fetch_latest_fanta_news
        news_text = fetch_latest_fanta_news()
        st.session_state["news_area"] = news_text
        st.session_state["news_fetched_at"] = time.time()
        with open(AUTO_NEWS_PATH, "w", encoding="utf-8") as f:
            f.write(news_text)
    except Exception:
        if "news_area" not in st.session_state:
            st.session_state["news_area"] = ""

if st.sidebar.button("🔄 Aggiorna News"):
    try:
        from news_scraper import fetch_latest_fanta_news
        news_text = fetch_latest_fanta_news()
        st.session_state["news_area"] = news_text
        st.session_state["news_fetched_at"] = time.time()
        with open(AUTO_NEWS_PATH, "w", encoding="utf-8") as f:
            f.write(news_text)
        st.rerun()
    except Exception as e:
        st.sidebar.warning(f"Errore: {e}")

news_input = st.sidebar.text_area(
    "Notizie Fantacalcio.it (modificabili):",
    key="news_area",
    height=150,
    help="Aggiornate automaticamente da fantacalcio.it/calciomercato ogni 30 minuti. Puoi modificare il testo prima di generare la rosa.",
)

QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
data_df = None

if os.path.exists(QUOTAZIONI_PATH):
    try:
        data_df = pd.read_csv(QUOTAZIONI_PATH)
        st.sidebar.caption(f"📋 Dataset Attivo: Listone Corrente ({len(data_df)} giocatori)")
    except Exception as e:
        st.sidebar.error(f"Errore caricamento cache: {e}")
elif os.path.exists(DEFAULT_DATA):
    try:
        data_df = pd.read_csv(DEFAULT_DATA)
        st.sidebar.caption("📋 Dataset Attivo: Demo 23/24")
    except Exception as e:
        st.sidebar.error(f"Errore: {e}")
else:
    st.sidebar.warning("Nessun dato. Carica un file.")

# ── AUTO-REFRESH INFORTUNI ────────────────────────────────────────────────────
# Aggiorna automaticamente se la cache è più vecchia di INJURIES_STALE_HOURS
st.sidebar.subheader("🏥 Infortuni")
if data_df is not None:
    if _cache_is_stale(INJURIES_CACHE_PATH, INJURIES_STALE_HOURS):
        with st.sidebar.status("🔄 Aggiornamento automatico infortuni...") as s:
            try:
                ok, n = fetch_and_save_injuries(data_df['nome'].tolist())
                s.update(
                    label=f"✅ Infortuni aggiornati ({n} eventi)" if ok else "⚠️ Nessun evento trovato",
                    state="complete", expanded=False,
                )
            except Exception:
                s.update(label="⚠️ Feed infortuni non raggiungibile", state="error", expanded=False)
    else:
        if st.sidebar.button("🔄 Forza aggiornamento infortuni"):
            with st.sidebar.status("Aggiornamento...") as s:
                try:
                    ok, n = fetch_and_save_injuries(data_df['nome'].tolist())
                    s.update(label=f"✅ {n} eventi trovati", state="complete", expanded=False)
                except Exception as e:
                    s.update(label=f"❌ {e}", state="error", expanded=False)

inj_cache = load_cached_injuries()
active_inj = [k for k, v in inj_cache.items() if v.get('modifier', 1.0) < 1.0]
st.sidebar.caption(
    f"Infortuni: {len(active_inj)} attivi — {_cache_age_str(INJURIES_CACHE_PATH)}"
    if inj_cache else "Infortuni: cache vuota"
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_opt, tab_copilot, tab_lineup, tab_top, tab_search, tab_vip, tab_inj, tab_cal, tab_bt = st.tabs([
    "🏆 Ottimizza Rosa",
    "⚡ Copilota Asta",
    "🗓️ Formazione Live",
    "📊 Top Scommesse",
    "🔍 Cerca Giocatore",
    "💎 VIP Radar",
    "🏥 Infortuni",
    "📅 Calendario",
    "🔬 Backtest",
])

# ── Tab 1: Ottimizza Rosa ─────────────────────────────────────────────────────
with tab_opt:
    st.caption(f"Strategia attiva: **{strategy_label}**")
    if st.button("🚀 Genera Formazione Ottimizzata"):
        if data_df is None:
            st.error("Carica un file di quotazioni prima.")
        else:
            with st.spinner(f"IA ({strategy_label}) in elaborazione…"):
                data_df.to_csv(INPUT_TEMP, index=False)
                # ← strategia passata qui
                df_pred = _get_cached_predictions(data_df, strategy).copy() if _get_cached_predictions(data_df, strategy) is not None else None

                if news_input:
                    from news_agent import apply_news_modifiers
                    df_pred = apply_news_modifiers(df_pred, news_input)
                    st.sidebar.info("💡 News applicate.")

                pred_path = os.path.join(DATA_DIR, "current_predictions.csv")
                df_pred.to_csv(pred_path, index=False)

                try:
                    team = optimize_team(pred_path, budget=budget, strategy=strategy)
                except ValueError as e:
                    st.error(f"Ottimizzazione impossibile: {e}")
                    team = None

            if team is None or team.empty:
                st.error("Nessuna formazione trovata. Aumenta il budget o carica più giocatori.")
            else:
                # Ordina per score_selezione prima di scegliere i titolari:
                # così i migliori N per ruolo diventano sempre titolari, non i primi nel CSV
                def _top(grp: pd.DataFrame, n: int) -> pd.DataFrame:
                    col = "score_selezione" if "score_selezione" in grp.columns else "previsione_ia"
                    return grp.nlargest(n, col)

                titolari = pd.concat([
                    _top(team[team["ruolo"] == "P"], 1),
                    _top(team[team["ruolo"] == "D"], 4),
                    _top(team[team["ruolo"] == "C"], 4),
                    _top(team[team["ruolo"] == "A"], 3),
                ])
                # Usa sempre previsione_ia (FM realistica) per le metriche mostrate all'utente
                punteggio_g = round(titolari["previsione_ia"].sum(), 2)
                punteggio_s = round(titolari["previsione_ia"].mean() * 11 * 38, 2)

                st.subheader(f"✅ Formazione — {strategy_label} (Budget: {budget})")
                col1, col2 = st.columns(2)
                with col1:
                    display_cols = [c for c in [
                        "nome", "ruolo", "costo_iniziale", "fanta_media",
                        "gol_pg", "assist_pg", "xg_pg",
                        "cambio_squadra", "upgrade_squadra",
                        "previsione_ia", "score_selezione",
                    ] if c in team.columns]
                    st.dataframe(
                        team[display_cols].rename(columns={
                            "previsione_ia": "FM prevista",
                            "score_selezione": "priorità strategia",
                            "cambio_squadra": "trasf.",
                            "upgrade_squadra": "upgrade",
                        }).round(2),
                        use_container_width=True,
                    )
                    st.caption(
                        "**FM prevista** = stima realistica del rendimento stagionale. "
                        "**Priorità strategia** = score usato dall'IA per scegliere la squadra "
                        f"con la strategia *{strategy_label}* — non è una previsione FM."
                    )
                    st.metric("⚽ Punteggio Previsto / giornata", punteggio_g)
                    st.metric("📅 Proiezione Stagionale (38 g.)", punteggio_s)
                    st.metric("💰 Costo Totale", int(team["costo_iniziale"].sum()))
                
                with col2:
                    st.subheader("📊 Confronto con Top Account")
                    report = run_performance_comparison(team)
                    st.table(report)
                    st.bar_chart(team.groupby("ruolo")["costo_iniziale"].sum())
                
                st.write("---")
                with st.expander("🤖 📋 Leggi i Report Scout dell'IA per i 25 scelti"):
                    from predict import build_player_explanation
                    for idx, row in team.sort_values(by=['ruolo', 'previsione_ia'], ascending=[False, False]).iterrows():
                        st.markdown(build_player_explanation(row))
                        st.write("---")

                # ── Spiegazione IA ──────────────────────────────────────────
                st.markdown("---")
                with st.expander("🧠 Perché l'IA ha scelto questi giocatori?", expanded=True):
                    spiegazione = _explain_team(titolari, team, strategy, strategy_label, budget)
                    st.markdown(spiegazione)

                    # Riserve
                    riserve = team[~team.index.isin(titolari.index)]
                    if not riserve.empty:
                        st.markdown("---")
                        st.markdown("**Riserve selezionate**")
                        riserve_cols = [c for c in ["nome", "ruolo", "costo_iniziale", "previsione_ia"] if c in riserve.columns]
                        st.dataframe(
                            riserve[riserve_cols].rename(columns={"previsione_ia": "prev. IA"}).round(2),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.caption(
                            "Le riserve completano il budget e soddisfano i vincoli di ruolo dell'asta. "
                            "L'IA le ha scelte massimizzando il valore residuo a parità di costo."
                        )

# ── Tab 2: Copilota Asta Live ─────────────────────────────────────────────────
with tab_copilot:
    st.subheader("⚡ Copilota Asta Live (Assistente in Tempo Reale)")
    st.caption("Guida la tua asta in diretta: calcola la max bid sostenibile per ogni giocatore e ricalcola il budget ad ogni acquisto.")
    
    if 'auction_state' not in st.session_state:
        st.session_state.auction_state = AuctionState(total_budget=budget)
        
    state: AuctionState = st.session_state.auction_state
    
    with st.expander("⚙️ Impostazioni Avversari (Opzionale)"):
        opps_str = st.text_input("Nomi Avversari (separati da virgola)", value=", ".join([t for t in state.teams if t != state.my_team_name]))
        if st.button("💾 Salva Nomi Avversari e Resetta"):
            opp_list = [o.strip() for o in opps_str.split(",") if o.strip()]
            st.session_state.auction_state = AuctionState(total_budget=budget, opponents=opp_list)
            st.rerun()

    # Metriche di riepilogo MIO TEAM
    st.markdown("### Il Mio Team")
    c_st1, c_st2, c_st3, c_st4 = st.columns(4)
    with c_st1:
        st.metric("💰 Budget Rimasto", f"{state.remaining_budget} / {budget} cr")
    with c_st2:
        st.metric("👥 Slot Coperti", f"{len(state.my_roster)} / 25", delta=f"{25 - len(state.my_roster)} da comprare")
    with c_st3:
        st.metric("⚽ Reparti", f"P:{state.role_counts['P']}/3 D:{state.role_counts['D']}/8 C:{state.role_counts['C']}/8 A:{state.role_counts['A']}/6")
    with c_st4:
        heat = state.get_market_heat()
        heat_label = "🔥 Inflazione" if heat > 1.05 else ("❄️ Deflazione" if heat < 0.95 else "⚖️ Equilibrato")
        st.metric("Market Heat", f"{heat:.2f}x", delta=heat_label, delta_color="inverse" if heat > 1.05 else "normal")

    st.markdown("---")

    # Registrazione acquisto
    if data_df is not None:
        if os.path.exists(INPUT_TEMP):
            with st.spinner("Inizializzazione IA Copilot..."):
                df_cop = _get_cached_predictions(data_df, strategy)
        else:
            df_cop = data_df.copy()
            
        df_cop = apply_real_market_logic(df_cop)
        
        # Integriamo CUI se il dataset ne è sprovvisto, o ricalcoliamolo al volo se serve
        if 'cui' not in df_cop.columns:
            try:
                from chaos_optimizer import compute_chaos_upside_index
                df_cop = compute_chaos_upside_index(df_cop)
            except Exception:
                pass
                
        bids_df = calculate_copilot_bids(df_cop, state, num_partecipanti)
        
        with st.expander("➕ Registra Giocatore Chiamato all'Asta", expanded=True):
            c_b1, c_b2, c_b3, c_b4 = st.columns([3, 2, 2, 2])
            with c_b1:
                unassigned_names = bids_df['nome'].tolist()
                player_sel = st.selectbox("Giocatore", unassigned_names)
            with c_b2:
                # Mostra tutti i team disponibili nel menu a tendina
                buyer = st.selectbox("Acquirente", list(state.teams.keys()))
            with c_b3:
                price_paid = st.number_input("Prezzo d'Asta (cr)", min_value=1, max_value=budget, value=1)
            with c_b4:
                st.write("")
                st.write("")
                if st.button("✅ Registra Acquisto"):
                    p_info = bids_df[bids_df['nome'] == player_sel].iloc[0].to_dict()
                    state.buy_player(p_info, price=int(price_paid), buyer=buyer)
                    st.success(f"Registrato {player_sel} a {price_paid} cr ({buyer})")
                    if buyer != state.my_team_name:
                        st.session_state.last_rival_buy = p_info
                    else:
                        st.session_state.last_rival_buy = None
                    if 'copilot_role_filter' in st.session_state:
                        st.session_state['saved_filter'] = st.session_state['copilot_role_filter']
                    st.rerun()

        # Alert di suggerimento dinamico (IA Copilot Reaction)
        last_rival_buy = getattr(st.session_state, 'last_rival_buy', None)
        if last_rival_buy is not None:
            l_ruolo = last_rival_buy.get('ruolo', 'C')
            l_nome = last_rival_buy.get('nome', '')
            
            # Cerca la migliore alternativa rimasta nello stesso ruolo per ME
            alt_df = bids_df[bids_df['ruolo'] == l_ruolo].sort_values('var_score', ascending=False)
            if not alt_df.empty:
                top_alt = alt_df.iloc[0]
                st.info(
                    f"🎯 **REAZIONE IA:** Un avversario ha appena preso **{l_nome}** ({l_ruolo}). "
                    f"Il miglior bersaglio tattico ancora libero per questo reparto è "
                    f"**{top_alt['nome']}** (FM Prevista: {top_alt.get('previsione_ia', 6.0):.2f}). "
                    f"Tieniti pronto a chiamarlo, non superare i **{int(top_alt['max_bid'])} crediti**!",
                    icon="⚡"
                )

        # Tabella consigliata per le prossime chiamate con filtro reparto
        st.subheader("💡 Prezzo Massimo Consigliato (Max Bid) per i prossimi svincolati")
        
        if 'saved_filter' in st.session_state and 'copilot_role_filter' not in st.session_state:
            st.session_state['copilot_role_filter'] = st.session_state['saved_filter']
            
        filter_role = st.radio("Filtra reparto:", ["Tutti", "P", "D", "C", "A"], horizontal=True, key="copilot_role_filter")
        
        bids_filtered = bids_df.copy()
        if filter_role != "Tutti":
            bids_filtered = bids_filtered[bids_filtered['ruolo'] == filter_role]

        disp_cols = [c for c in [
            "nome", "ruolo", "squadra", "costo_iniziale",
            "previsione_ia", "var_score", "target_bid", "max_bid", "cui", "cui_label", "bonus_rigorista"
        ] if c in bids_filtered.columns]
        
        col_renames = {
            "costo_iniziale": "Listone cr",
            "previsione_ia": "FM Prevista",
            "var_score": "VAR",
            "target_bid": "Target Bid (cr)",
            "max_bid": "MAX BID CONSIGLIATO (cr)",
            "cui": "Chaos Index",
            "cui_label": "Profilo (CUI)",
            "bonus_rigorista": "Rigore Bonus"
        }
        
        df_view = bids_filtered[disp_cols].rename(columns=col_renames).head(100)
        
        # Funzioni di stile per Pandas
        def highlight_roles(val):
            colors = {
                'P': 'background-color: #f39c12; color: #1a1a1a; font-weight: 800; text-align: center;',
                'D': 'background-color: #27ae60; color: #ffffff; font-weight: 800; text-align: center;',
                'C': 'background-color: #2980b9; color: #ffffff; font-weight: 800; text-align: center;',
                'A': 'background-color: #c0392b; color: #ffffff; font-weight: 800; text-align: center;'
            }
            return colors.get(str(val).upper(), '')

        def highlight_cui(val):
            val_str = str(val)
            if 'Boom' in val_str: return 'color: #e74c3c; font-weight:bold;'
            if 'Costante' in val_str: return 'color: #2ecc71; font-weight:bold;'
            if 'Azzardo' in val_str: return 'color: #f39c12; font-style: italic;'
            if 'Flop' in val_str: return 'color: #95a5a6;'
            return ''
            
        styled_df = df_view.style
        if 'ruolo' in df_view.columns:
            styled_df = styled_df.map(highlight_roles, subset=['ruolo'])
        if 'Profilo (CUI)' in df_view.columns:
            styled_df = styled_df.map(highlight_cui, subset=['Profilo (CUI)'])
            
        if 'FM Prevista' in df_view.columns:
            styled_df = styled_df.background_gradient(cmap='YlGn', subset=['FM Prevista'])
        if 'MAX BID CONSIGLIATO (cr)' in df_view.columns:
            styled_df = styled_df.background_gradient(cmap='OrRd', subset=['MAX BID CONSIGLIATO (cr)'])

        format_dict = {}
        if 'FM Prevista' in df_view.columns: format_dict['FM Prevista'] = "{:.2f}"
        if 'VAR' in df_view.columns: format_dict['VAR'] = "+{:.2f}"
        if 'Chaos Index' in df_view.columns: format_dict['Chaos Index'] = "{:.2f}"
        if 'Listone cr' in df_view.columns: format_dict['Listone cr'] = "{:.0f}"
        if 'Target Bid (cr)' in df_view.columns: format_dict['Target Bid (cr)'] = "{:.0f}"
        if 'MAX BID CONSIGLIATO (cr)' in df_view.columns: format_dict['MAX BID CONSIGLIATO (cr)'] = "{:.0f}"
        
        styled_df = styled_df.format(format_dict)

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=500
        )
        
        with st.expander("📊 Situazione Avversari", expanded=False):
            opps_data = []
            for t_name, t_data in state.teams.items():
                if t_name == state.my_team_name: continue
                roster_len = len(t_data['roster'])
                rc = t_data['role_counts']
                rs = t_data['role_spent']
                
                # Deduce focus
                focus = "Bilanciato"
                if rs['A'] > t_data['budget'] * 0.45 or rc['A'] >= 4: focus = "Attacco Pesante"
                elif rc['D'] >= 5: focus = "Difesa Completa"
                elif t_data['budget'] > sum(other['budget'] for other in state.teams.values()) / max(1, len(state.teams)): focus = "Accumulatore"

                opps_data.append({
                    "Avversario": t_name,
                    "Budget Rimasto": t_data['budget'],
                    "Slot Coperti": f"{roster_len}/25",
                    "Reparti": f"P:{rc['P']} D:{rc['D']} C:{rc['C']} A:{rc['A']}",
                    "Strategia Dedotta": focus
                })
            if opps_data:
                st.table(pd.DataFrame(opps_data))

        if state.my_roster:
            st.subheader("📋 La Mia Rosa Acquistata")
            st.dataframe(pd.DataFrame(state.my_roster), use_container_width=True)
    else:
        st.warning("Carica un file di quotazioni per avviare il Copilota Asta Live.")


# ── Tab 3: Top Scommesse ──────────────────────────────────────────────────────
# ── Tab 2b: Formazione Live ───────────────────────────────────────────────────
with tab_lineup:
    st.subheader("🗓️ Copilota Schieramento Settimanale (Game Theory)")
    st.markdown("Massimizza la probabilità di vittoria calcolando difficoltà match e copertura tattica contro l'avversario.")
    
    if st.session_state.auction_state.my_roster:
        st.success(f"Rosa attuale: {len(st.session_state.auction_state.my_roster)} giocatori trovati dal Copilota Asta.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Tua Rosa")
            my_roster_df = pd.DataFrame(st.session_state.auction_state.my_roster)
            st.dataframe(my_roster_df[['nome', 'ruolo', 'squadra', 'previsione_ia']], use_container_width=True)
            
        with c2:
            st.markdown("### Rosa Avversario (Opzionale)")
            opp_text = st.text_area("Incolla i giocatori del tuo avversario (uno per riga) per attivare l'Hedging Strategico:", height=150)
            
        if st.button("🧠 Calcola Formazione Ottimizzata"):
            with st.spinner("Risoluzione MILP in corso..."):
                try:
                    import lineup_copilot
                    import importlib
                    importlib.reload(lineup_copilot)
                    from lineup_copilot import optimize_weekly_lineup
                    
                    # Preparazione df avversario
                    opp_df = None
                    if opp_text.strip():
                        opp_names = [n.strip().lower() for n in opp_text.split('\n') if n.strip()]
                        if data_df is not None:
                            # Troviamo le squadre dei giocatori avversari
                            data_df_lower = data_df.copy()
                            if 'nome' in data_df_lower.columns:
                                data_df_lower['nome_low'] = data_df_lower['nome'].str.lower()
                                opp_df = data_df_lower[data_df_lower['nome_low'].isin(opp_names)]
                    
                    res = optimize_weekly_lineup(my_roster_df, opp_df)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        st.success(f"Formazione ideale trovata! Modulo: **{res['modulo']}** — Punti Proiettati: **{res['totale_proiettato']}**")
                        
                        titolari = res['titolari']
                        panchina = res['panchina']
                        
                        st.markdown("### 🏟️ Titolari")
                        st.dataframe(titolari[['ruolo', 'nome', 'squadra', 'fm_attesa', 'match_multiplier', 'punti_proiettati']], use_container_width=True)
                        
                        st.markdown("### 🪑 Panchina (In ordine di subentro)")
                        st.dataframe(panchina[['ruolo', 'nome', 'squadra', 'fm_attesa', 'punti_proiettati']], use_container_width=True)
                except Exception as e:
                    st.error(f"Errore nel calcolo: {e}")
    else:
        st.warning("Devi prima acquistare giocatori nel **Copilota Asta Live** per poter calcolare la formazione!")

# ── Tab 3: Top Scommesse ──────────────────────────────────────────────────────
with tab_top:
    st.subheader(f"📊 Top Scommesse Qualità/Prezzo per Reparto — {strategy_label}")
    if data_df is None:
        st.warning("Carica un file di quotazioni.")
    else:
        with st.spinner("Calcolo in corso..."):
            data_df.to_csv(INPUT_TEMP, index=False)
            df_s = _get_cached_predictions(data_df, strategy)

        # Rapporto qualità/prezzo normalizzato per reparto per evitare che 1cr monopolizzi
        df_s["convenienza"] = df_s["previsione_ia"] / (df_s["costo_iniziale"].clip(lower=1) + 2)

        show_cols = [c for c in [
            "nome", "ruolo", "costo_iniziale",
            "gol_pg", "assist_pg", "xg_pg",
            "titolarita_pct", "previsione_ia", "convenienza",
        ] if c in df_s.columns]

        t_p, t_d, t_c, t_a = st.tabs(["🧤 Portieri", "🛡️ Difensori", "⚙️ Centrocampisti", "⚽ Attaccanti"])
        
        role_map = [("P", t_p, 3), ("D", t_d, 8), ("C", t_c, 8), ("A", t_a, 6)]
        for r_code, tab_obj, top_n in role_map:
            with tab_obj:
                sub_r = df_s[df_s["ruolo"] == r_code].nlargest(top_n, "convenienza")
                st.dataframe(
                    sub_r[show_cols].rename(columns={
                        "previsione_ia": "prev. IA",
                        "convenienza": "qualità/prezzo",
                        "costo_iniziale": "costo (cr)",
                        "titolarita_pct": "titolarità %",
                    }).round(3),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Tab 3: Cerca Giocatore ────────────────────────────────────────────────────
with tab_search:
    st.subheader("🔍 Cerca Giocatore")
    if data_df is None:
        st.warning("Carica un file di quotazioni.")
    else:
        query = st.text_input("Nome (anche parziale)", placeholder="es. Lautaro, Theo, Barella...")
        if query.strip():
            data_df.to_csv(INPUT_TEMP, index=False)
            # ← strategia passata qui
            df_all = _get_cached_predictions(data_df, strategy)
            mask = df_all["nome"].str.contains(query.strip(), case=False, na=False)
            risultati = df_all[mask]
            if risultati.empty:
                st.info(f"Nessun giocatore trovato per '{query}'.")
            else:
                show_cols = [c for c in [
                    "nome", "ruolo", "costo_iniziale", "fanta_media",
                    "gol_pg", "assist_pg", "xg_pg", "titolarita_pct",
                    "cambio_squadra", "upgrade_squadra", "tipo_prestito",
                    "previsione_ia",
                ] if c in risultati.columns]
                st.dataframe(
                    risultati[show_cols].rename(columns={"previsione_ia": "prev. IA"}).round(3),
                    use_container_width=True,
                )

# ── Tab 4: VIP Radar ─────────────────────────────────────────────────────────
with tab_vip:
    st.subheader("💎 VIP Radar — Valore Intrinseco Predittivo")
    st.markdown(
        "Il **VIP** codifica la conoscenza calcistica che il modello ML non vede: "
        "posizione tattica reale, performance per 90 minuti con poca campione, "
        "crescita esponenziale dei giovani under-22."
    )

    col_vip1, col_vip2 = st.columns([2, 1])
    with col_vip1:
        if data_df is None:
            st.warning("Carica un file di quotazioni.")
        else:
            with st.spinner("Calcolo VIP in corso..."):
                data_df.to_csv(INPUT_TEMP, index=False)
                df_vip = _get_cached_predictions(data_df, strategy)

            vip_cols = [c for c in [
                "nome", "ruolo", "costo_iniziale",
                "vip_position", "vip_tpm", "vip_p90", "vip_youth", "vip_total",
                "previsione_ia", "vip_score",
            ] if c in df_vip.columns]

            if "vip_total" not in df_vip.columns:
                st.info("VIP non disponibile — installa il modulo vip.py.")
            else:
                vip_filter = st.selectbox(
                    "Filtra per ruolo", ["Tutti", "P", "D", "C", "A"], key="vip_ruolo"
                )
                df_show = df_vip.copy()
                if vip_filter != "Tutti":
                    df_show = df_show[df_show["ruolo"] == vip_filter]

                top_n = st.slider("Mostra top N giocatori per VIP", 5, 50, 20, key="vip_n")
                df_show = df_show.nlargest(top_n, "vip_total")

                def _vip_color(val):
                    if isinstance(val, float):
                        if val >= 0.30:
                            return "background-color:#1a472a;color:white"
                        if val >= 0.15:
                            return "background-color:#2d6a4f;color:white"
                        if val >= 0.05:
                            return "background-color:#52b788;color:black"
                        if val < 0:
                            return "background-color:#e63946;color:white"
                    return ""

                st.dataframe(
                    df_show[vip_cols].rename(columns={
                        "vip_position": "posizione",
                        "vip_tpm":   "TPM",
                        "vip_p90":   "P90",
                        "vip_youth": "Youth",
                        "vip_total": "VIP tot",
                        "vip_score": "score VIP",
                        "previsione_ia": "prev. IA",
                        "costo_iniziale": "costo",
                    }).style.map(
                        _vip_color,
                        subset=[c for c in ["TPM", "P90", "Youth", "VIP tot"] if c in
                                df_show.rename(columns={"vip_tpm":"TPM","vip_p90":"P90","vip_youth":"Youth","vip_total":"VIP tot"}).columns]
                    ).format(precision=3),
                    use_container_width=True,
                )
                st.caption(
                    "**TPM** = Tactical Position Modifier (bonus ruolo tattico avanzato) · "
                    "**P90** = Discovery factor metriche bayesiane per 90 min · "
                    "**Youth** = coefficiente crescita under-22 · "
                    "**VIP tot** = somma clippata · "
                    "**score VIP** = prev. IA × (1 + VIP tot)"
                )

    with col_vip2:
        st.success("✨ **Motore VIP Completamente Automatico**")
        st.markdown(
            "Il VIP Radar ora scansiona l'intero database in totale autonomia. "
            "L'algoritmo rileva da solo i giovani talenti U22 incrociando l'età con i trend di minutaggio, "
            "e deduce la posizione tattica reale (es. *Quinto d'attacco* per i difensori goleador) "
            "utilizzando l'inferenza statistica sui tassi di Gol e Assist P90.\n\n"
            "Non c'è più bisogno di alcun inserimento manuale!"
        )

CURRENT_YEAR = 2026  # usato nel tab VIP

# ── Tab 5: Infortuni ──────────────────────────────────────────────────────────
with tab_inj:
    st.subheader("🏥 Stato Infortuni e Sospensioni")
    st.caption(f"Aggiornati automaticamente ogni {INJURIES_STALE_HOURS}h · Ultimo aggiornamento: {_cache_age_str(INJURIES_CACHE_PATH)}")
    if data_df is None:
        st.warning("Carica un file di quotazioni per vedere gli infortuni.")
    else:
        inj_cache = load_cached_injuries()
        if not inj_cache:
            st.info("Nessun dato. L'aggiornamento automatico avverrà al prossimo caricamento.")
        else:
            summary = get_injury_summary(data_df['nome'].tolist())
            if not summary:
                st.success("✅ Nessun infortunio rilevato per i giocatori in lista.")
            else:
                inj_df = pd.DataFrame(summary)[
                    ['nome', 'tipo', 'modifier', 'duration_days', 'testo']
                ].rename(columns={
                    'tipo': 'evento', 'modifier': 'impatto',
                    'duration_days': 'giorni stimati', 'testo': 'notizia',
                })

                def color_row(row):
                    if row['impatto'] < 0.5:
                        return ['background-color: #ffcccc'] * len(row)
                    if row['impatto'] < 0.85:
                        return ['background-color: #fff3cc'] * len(row)
                    return [''] * len(row)

                st.dataframe(inj_df.style.apply(color_row, axis=1), use_container_width=True)
                st.caption("🔴 Rosso = infortuno grave  🟡 Giallo = stop breve/panchina  ⬜ Bianco = recupero/bonus")


# ── Tab: Calendario e Portieri ────────────────────────────────────────────────
with tab_cal:
    st.subheader("📅 Analisi Calendario e Incroci Portieri")
    st.markdown("Questa sezione analizza il calendario della Serie A per trovare la difficoltà delle prossime partite per ogni squadra e calcolare le migliori coppie di portieri da acquistare all'asta.")
    
    if data_df is None:
        st.warning("Carica un file di quotazioni per vedere i dati.")
    else:
        from calendar_analyzer import fetch_calendar, get_all_teams_difficulty, find_best_gk_pairings
        
        with st.spinner("Scaricamento calendario in corso..."):
            calendar = fetch_calendar()
            
        if not calendar:
            st.error("Impossibile scaricare il calendario in questo momento.")
        else:
            col_cal1, col_cal2 = st.columns([1, 2])
            
            with col_cal1:
                st.markdown("### 📊 Difficoltà Squadre")
                num_matches = st.slider("Numero di prossime partite da analizzare:", min_value=1, max_value=38, value=38)
                start_match = st.number_input("Partita di partenza (Giornata):", min_value=1, max_value=38, value=1)
                
                diffs = get_all_teams_difficulty(calendar, from_matchday=start_match, num_matches=num_matches)
                diff_df = pd.DataFrame(list(diffs.items()), columns=["Squadra", "Difficoltà (1-5)"])
                diff_df = diff_df.sort_values(by="Difficoltà (1-5)").reset_index(drop=True)
                
                def color_diff(val):
                    if val <= 2.5: return 'background-color: #d4edda; color: black;' # verde
                    if val >= 3.8: return 'background-color: #f8d7da; color: black;' # rosso
                    return ''
                    
                st.dataframe(diff_df.style.map(color_diff, subset=['Difficoltà (1-5)']), use_container_width=True)
                st.caption("Verde = Calendario Facile, Rosso = Calendario Difficile")
                
            with col_cal2:
                st.markdown("### 🧤 Migliori Coppie Portieri")
                gk_matches = st.slider("Analizza incroci per le prossime N giornate:", min_value=1, max_value=38, value=38)
                
                                # Trova colonna squadra
                sq_col = next((c for c in data_df.columns if c.lower() in ('squadra', 'sq', 'team')), None)
                squadre_presenti = data_df[sq_col].dropna().unique().tolist() if sq_col else []
                
                from calendar_analyzer import TEAM_STRENGTH
                squadre_pulite = list(TEAM_STRENGTH.keys())
                    
                pairings = find_best_gk_pairings(calendar, squadre_pulite, from_matchday=start_match, num_matches=gk_matches)
                
                # Sostituisci il nome della squadra con il nome del portiere titolare
                team_to_gks = {}
                ruolo_col = next((c for c in data_df.columns if c.lower() in ('ruolo', 'r', 'role')), None)
                nome_col = next((c for c in data_df.columns if c.lower() in ('nome', 'giocatore', 'player', 'n')), None)
                
                if sq_col and ruolo_col and nome_col:
                    # Usa i dati aggiornati caricati dall'utente per la stagione corrente
                    for sq in squadre_pulite:
                        gks = data_df[(data_df[sq_col].astype(str).str.strip().str.upper() == sq.strip().upper()) & (data_df[ruolo_col].astype(str).str.strip().str.upper().str.startswith("P"))]
                        if not gks.empty:
                            sort_col = "costo_iniziale" if "costo_iniziale" in data_df.columns else "fanta_media" if "fanta_media" in data_df.columns else None
                            if sort_col:
                                gks = gks.sort_values(by=sort_col, ascending=False)
                            top_gk = gks.iloc[0][nome_col]
                            team_to_gks[sq] = f"{top_gk} ({sq})"
                        else:
                            team_to_gks[sq] = sq
                else:
                    for sq in squadre_pulite:
                        team_to_gks[sq] = sq
                
                for p in pairings:
                    p["team1"] = team_to_gks.get(p["team1"], p["team1"])
                    p["team2"] = team_to_gks.get(p["team2"], p["team2"])

                pair_df = pd.DataFrame(pairings)
                pair_df = pair_df.rename(columns={
                    "team1": "Portiere 1 (Squadra)",
                    "team2": "Portiere 2 (Squadra)",
                    "avg_difficulty": "Difficoltà Incrociata",
                    "alternation_pct": "% Alternanza Casa/Trasferta"
                })
                
                st.dataframe(pair_df.head(20), use_container_width=True)
                st.caption("Mostra le migliori 20 coppie in base alla difficoltà della partita più facile in ogni giornata. Una % Alternanza alta indica che quasi sempre uno gioca in casa e l'altro in trasferta.")


# ── Tab 6: Backtest IA ────────────────────────────────────────────────────────
with tab_bt:
    st.subheader("🔬 Backtest Causal-Temporale (Zero Data Leakage)")
    st.markdown(
        "Valuta le prestazioni reali dell'IA: il sistema addestra il modello **esclusivamente sui dati "
        "della stagione precedente**, sceglie la rosa col solver senza conoscere i voti della nuova stagione, "
        "e infine ne confronta i risultati con la **Squadra Oracle** (la rosa ideale teorica col senno di poi)."
    )

    c_bt1, c_bt2 = st.columns(2)
    with c_bt1:
        bt_budget = st.number_input(
            "Budget asta backtest", min_value=100, max_value=1000, value=500, step=50, key="bt_budget_input"
        )
    with c_bt2:
        bt_strat_label = st.selectbox(
            "Strategia IA da testare",
            ["🏆 Listone Master (11 Top)", "Conservativa (MV)", "Aggressiva (Bonus)", "Scommesse (Hype)", "VIP Premium"],
            key="bt_strat_input"
        )
    bt_strategy = _strategy_key(bt_strat_label)

    st.markdown("---")

    # ── Upload 1: stagione precedente (input IA) ──────────────────────────────
    st.markdown("#### 1. Stagione precedente — input dell'IA (Pre-campionato)")
    st.caption(
        "Il file da cui l'IA impara: quotazioni/statistiche della stagione "
        "**prima** di quella che vuoi predire (es. 2022/23)."
    )
    bt_stats_file = st.file_uploader(
        "Carica file stagione precedente (.xlsx o .csv)",
        type=["xlsx", "csv"],
        key="bt_stats_upload",
    )
    bt_stats_df = None
    if bt_stats_file:
        try:
            import tempfile as _tmpmod
            suffix = ".xlsx" if bt_stats_file.name.endswith(".xlsx") else ".csv"
            with _tmpmod.NamedTemporaryFile(suffix=suffix, delete=False, dir=DATA_DIR) as _f:
                _f.write(bt_stats_file.getbuffer())
                _tmp_stats = _f.name
            from importer import import_real_quotazioni
            bt_stats_df = import_real_quotazioni(_tmp_stats)
            os.unlink(_tmp_stats)
            if bt_stats_df is not None and not bt_stats_df.empty:
                st.success(f"✅ {len(bt_stats_df)} giocatori caricati — stagione precedente")
                with st.expander("Anteprima"):
                    st.dataframe(bt_stats_df.head(10), use_container_width=True)
            else:
                st.error("File non riconosciuto. Verifica che contenga nome, ruolo, costo e FM.")
                bt_stats_df = None
        except Exception as _e:
            st.error(f"Errore lettura file: {_e}")
            bt_stats_df = None

    st.markdown("---")

    # ── Upload 2: stagione da predire (ground truth) ──────────────────────────
    st.markdown("#### 2. Stagione da predire — risultati reali (Fine campionato)")
    st.caption(
        "Il file con i risultati **veri**: quotazioni e FantaMedia della stagione conclusa."
    )
    bt_target_file = st.file_uploader(
        "Carica file stagione da predire (.xlsx o .csv)",
        type=["xlsx", "csv"],
        key="bt_target_upload",
    )
    bt_target_df = None
    if bt_target_file:
        try:
            suffix = ".xlsx" if bt_target_file.name.endswith(".xlsx") else ".csv"
            with _tmpmod.NamedTemporaryFile(suffix=suffix, delete=False, dir=DATA_DIR) as _f:
                _f.write(bt_target_file.getbuffer())
                _tmp_target = _f.name
            from importer import import_real_quotazioni
            bt_target_df = import_real_quotazioni(_tmp_target)
            os.unlink(_tmp_target)
            if bt_target_df is not None and not bt_target_df.empty:
                st.success(f"✅ {len(bt_target_df)} giocatori caricati — stagione da predire")
                with st.expander("Anteprima"):
                    st.dataframe(bt_target_df.head(10), use_container_width=True)
            else:
                st.error("File non riconosciuto. Verifica che contenga nome, ruolo, costo e FM.")
                bt_target_df = None
        except Exception as _e:
            st.error(f"Errore lettura file: {_e}")
            bt_target_df = None

    if bt_target_df is None:
        st.caption("ℹ️ Nessun file caricato — verrà usato il dataset di storico predefinito.")

    st.markdown("---")

    # ── Esegui backtest ───────────────────────────────────────────────────────
    can_run = bt_target_df is not None or True  # demo sempre disponibile
    if st.button("▶️ Esegui Backtest", disabled=False):
        with st.spinner("Simulazione in corso… (20-40 sec)"):
            try:
                from backtest import run_ia_backtest
                result = run_ia_backtest(
                    budget=bt_budget,
                    target_df=bt_target_df,
                    stats_df=bt_stats_df,
                )
            except Exception as e:
                st.error(f"Errore: {e}")
                result = None

        if result:
            m = result.get('metrics', {})
            eff = result.get('efficiency', round(result.get('ai_score_real', 0) / max(1.0, result.get('oracle_score', 1)) * 100, 1))

            if not result.get('has_real_fm', True):
                st.warning(
                    "⚠️ Il file caricato per la **Stagione da predire** non contiene i voti/FM reali della stagione conclusa "
                    "(è un file di quotazioni pre-stagione). Per un backtest completo con accuratezza reale, carica il file "
                    "di una stagione già conclusa contenente le FantaMedie finali."
                )

            st.subheader(f"📊 Risultati Backtest — Strategia: {bt_strat_label}")
            
            n_comuni = len(result['df_comparison'])

            if bt_stats_df is None:
                st.warning(
                    "Hai usato la modalità demo (nessun file stagione precedente caricato). "
                    "I risultati sono poco significativi: carica i due file per un backtest reale."
                )

            st.subheader("Accuratezza previsioni")
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{m['rmse']} FM", help="Errore medio assoluto delle previsioni FM. Sotto 1.0 è buono.")
            c2.metric("Correlazione", f"{m['corr']}", help="Quanto l'IA ordina correttamente i giocatori. Sopra 0.7 è buono.")
            c3.metric("Top-10 Accuracy", f"{m.get('top10_accuracy', 0)*100:.0f}%", help="% dei 10 migliori giocatori reali che l'IA aveva identificato.")
            st.caption(f"Giocatori in comune tra i due file: {n_comuni}")

            st.subheader("Confronto squadre")
            c4, c5, c6 = st.columns(3)
            c4.metric("FM Oracle", result['oracle_score'], help="Punteggio massimo teorico — squadra perfetta costruita con i risultati già noti.")
            c5.metric("FM AI (reale)", result['ai_score_real'], help="Punteggio FM che la squadra scelta dall'IA avrebbe fatto davvero.")
            eff = round(result['ai_score_real'] / result['oracle_score'] * 100, 1) if result['oracle_score'] else 0
            c6.metric("Efficienza AI", f"{eff}%", delta=f"{result['ai_score_real'] - result['oracle_score']:.2f}", help="Quanto si avvicina all'Oracle. Sopra 90% è ottimo.")

            col_or, col_ai = st.columns(2)
            with col_or:
                st.markdown("**Squadra ORACLE** *(risultati già noti)*")
                _cols_or = [c for c in ['nome', 'ruolo', 'fanta_media', 'costo_iniziale'] if c in result['oracle_team'].columns]
                st.dataframe(result['oracle_team'][_cols_or].round(2), use_container_width=True)
            with col_ai:
                st.markdown("**Squadra AI** *(scelta pre-stagione)*")
                _cols_ai = [c for c in ['nome', 'ruolo', 'previsione_ia', 'fanta_media', 'costo_iniziale'] if c in result['ai_team'].columns]
                st.dataframe(
                    result['ai_team'][_cols_ai]
                    .rename(columns={'previsione_ia': 'prev. IA', 'fanta_media': 'FM reale'})
                    .round(2),
                    use_container_width=True,
                )

            st.subheader("Previsto vs Reale")
            df_comp = result['df_comparison'].copy()
            fm_real_col = 'fanta_media_y' if 'fanta_media_y' in df_comp.columns else 'fanta_media'
            if 'previsione_ia' in df_comp.columns and fm_real_col in df_comp.columns:
                df_chart = df_comp.rename(columns={'previsione_ia': 'Previsto', fm_real_col: 'Reale'})
                st.scatter_chart(
                    df_chart,
                    x='Previsto', y='Reale',
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
if fc_excel_file is not None:
    # removed
    st.info(f"✨ Dati caricati: **{fc_excel_file.name}**")
