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
    if "Listone" in label:
        return "listone"
    if "Conserv" in label:
        return "conservativa"
    if "Agg" in label:
        return "aggressiva"
    return "scommesse"


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

st.markdown("""
<div class="hero-header">
    <div class="hero-title">🤖 FantaBot AI 2026/27 — Market Master</div>
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
    "Strategia IA",
    ["🏆 Listone Master (11 Top)", "Conservativa (MV)", "Aggressiva (Bonus)", "Scommesse (Hype)"],
    help=(
        "🏆 Listone Master: 88% budget sui 11 Super-Top Titolari + 14 riserve 1cr a titolarità garantita.\n"
        "Conservativa: si fida della fanta_media storica — bassa varianza.\n"
        "Aggressiva: premia chi fa gol e assist — alta volatilità.\n"
        "Scommesse: premia i sottovalutati (qualità/prezzo) — massima sorpresa."
    ),
)
strategy = _strategy_key(strategy_label)

# ── Stats Fantacalcio.it ──────────────────────────────────────────────────────
st.sidebar.subheader("📡 Stats Fantacalcio.it")

_fc_cache_path = os.path.join(DATA_DIR, "fantacalcio_stats_cache.csv")

# Opzione A: login automatico con credenziali
with st.sidebar.expander("🔑 Login automatico (email + password)"):
    st.caption(
        "Il sito Fantacalcio.it è una SPA (app JavaScript): i dati non sono nel HTML pubblico. "
        "Inserisci le tue credenziali per scaricare automaticamente l'Excel ufficiale."
    )
    fc_email    = st.text_input("Email", key="fc_email", placeholder="tua@email.it")
    fc_password = st.text_input("Password", key="fc_password", type="password")
    if st.button("🔄 Scarica Stats (login auto)"):
        with st.sidebar.status("Login + download da Fantacalcio.it...") as s:
            try:
                ok, msg = fetch_and_save_stats(
                    email=st.session_state.get("fc_email", ""),
                    password=st.session_state.get("fc_password", ""),
                )
                s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)
            except Exception as e:
                s.update(label=f"❌ {e}", state="error")

# Opzione B: upload manuale Excel (raccomandato se il login automatico non funziona)
with st.sidebar.expander("📂 Upload Excel manuale (alternativa)"):
    st.caption(
        "1. Vai su **fantacalcio.it** nel tuo browser e accedi\n"
        "2. Apri la pagina **Statistiche Serie A**\n"
        "3. Clicca il pulsante **Excel** per scaricare il file\n"
        "4. Carica il file qui sotto"
    )
    fc_excel_file = st.file_uploader(
        "Carica Excel Fantacalcio.it", type=["xlsx", "xls"], key="fc_excel_upload"
    )
    if fc_excel_file is not None:
        with st.sidebar.status("Importazione Excel...") as s:
            try:
                ok, msg = save_manual_excel(fc_excel_file.getvalue())
                s.update(label=msg if ok else f"❌ {msg}", state="complete" if ok else "error", expanded=False)
            except Exception as e:
                s.update(label=f"❌ {e}", state="error")

cached_stats = load_cached_stats()
if cached_stats is not None:
    st.sidebar.caption(
        f"Stats: {len(cached_stats)} giocatori — aggiornato {_cache_age_str(_fc_cache_path)}"
    )
else:
    st.sidebar.caption("⚠️ Stats: nessuna cache — usa login auto o carica l'Excel manualmente")

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

# ── File uploader + caricamento dati ─────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader("Carica Quotazioni (.csv, .xlsx)", type=["csv", "xlsx"])

QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")

data_df = None
if uploaded_file is not None:
    ext = os.path.splitext(uploaded_file.name)[1]
    tmp = os.path.join(BASE_DIR, f"temp_quotazioni{ext}")
    with open(tmp, "wb") as f:
        f.write(uploaded_file.getbuffer())
    data_df = import_real_quotazioni(tmp)
    if data_df is not None:
        data_df.to_csv(QUOTAZIONI_PATH, index=False)
        st.sidebar.success(f"✅ {len(data_df)} giocatori caricati (Stagione Corrente).")
        with st.expander("👀 Anteprima Listone Caricato"):
            st.dataframe(data_df.head(10))
    else:
        st.sidebar.warning("⚠️ File non leggibile. Uso dati in cache come fallback.")
        if os.path.exists(QUOTAZIONI_PATH):
            data_df = pd.read_csv(QUOTAZIONI_PATH)
        elif os.path.exists(DEFAULT_DATA):
            data_df = pd.read_csv(DEFAULT_DATA)
else:
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
tab_opt, tab_copilot, tab_lineup, tab_top, tab_search, tab_vip, tab_inj, tab_bt = st.tabs([
    "🏆 Ottimizza Rosa",
    "⚡ Copilota Asta",
    "🗓️ Formazione Live",
    "📊 Top Scommesse",
    "🔍 Cerca Giocatore",
    "💎 VIP Radar",
    "🏥 Infortuni",
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
                df_pred = train_prediction_model(INPUT_TEMP, strategy=strategy)

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
    
    # Metriche di riepilogo
    c_st1, c_st2, c_st3, c_st4 = st.columns(4)
    with c_st1:
        st.metric("💰 Budget Rimasto", f"{state.remaining_budget} / {budget} cr")
    with c_st2:
        st.metric("👥 Slot Coperti", f"{len(state.my_roster)} / 25", delta=f"{25 - len(state.my_roster)} da comprare")
    with c_st3:
        st.metric("⚽ Reparti", f"P:{state.role_counts['P']}/3 D:{state.role_counts['D']}/8 C:{state.role_counts['C']}/8 A:{state.role_counts['A']}/6")
    with c_st4:
        if st.button("🔄 Resetta Asta"):
            st.session_state.auction_state = AuctionState(total_budget=budget)
            st.rerun()

    st.markdown("---")

    # Registrazione acquisto
    if data_df is not None:
        if os.path.exists(INPUT_TEMP):
            with st.spinner("Inizializzazione IA Copilot..."):
                df_cop = train_prediction_model(INPUT_TEMP, strategy=strategy)
        else:
            df_cop = data_df.copy()
            
        df_cop = apply_real_market_logic(df_cop)
        bids_df = calculate_copilot_bids(df_cop, state, num_partecipanti)
        
        with st.expander("➕ Registra Giocatore Chiamato all'Asta", expanded=True):
            c_b1, c_b2, c_b3, c_b4 = st.columns([3, 2, 2, 2])
            with c_b1:
                unassigned_names = bids_df['nome'].tolist()
                player_sel = st.selectbox("Giocatore", unassigned_names)
            with c_b2:
                buyer = st.selectbox("Acquirente", ["ME", "RIVALE"])
            with c_b3:
                price_paid = st.number_input("Prezzo d'Asta (cr)", min_value=1, max_value=budget, value=1)
            with c_b4:
                st.write("")
                st.write("")
                if st.button("✅ Registra Acquisto"):
                    p_info = bids_df[bids_df['nome'] == player_sel].iloc[0].to_dict()
                    state.buy_player(p_info, price=int(price_paid), buyer=buyer)
                    st.success(f"Registrato {player_sel} a {price_paid} cr ({buyer})")
                    if buyer == "RIVALE":
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
            
            # Cerca la migliore alternativa rimasta nello stesso ruolo
            alt_df = bids_df[bids_df['ruolo'] == l_ruolo].sort_values('var_score', ascending=False)
            if not alt_df.empty:
                top_alt = alt_df.iloc[0]
                st.info(
                    f"🎯 **REAZIONE IA:** Un rivale ha appena preso **{l_nome}** ({l_ruolo}). "
                    f"Il miglior bersaglio tattico ancora libero per questo reparto è "
                    f"**{top_alt['nome']}** (FM Prevista: {top_alt['previsione_ia']:.2f}). "
                    f"Tieniti pronto a chiamarlo, ma non superare i **{int(top_alt['max_bid'])} crediti**!",
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
            "previsione_ia", "var_score", "target_bid", "max_bid", "bonus_rigorista"
        ] if c in bids_filtered.columns]
        st.dataframe(
            bids_filtered[disp_cols].rename(columns={
                "costo_iniziale": "Listone cr",
                "previsione_ia": "FM Prevista",
                "var_score": "VAR Marginalità",
                "target_bid": "Target Bid (cr)",
                "max_bid": "MAX BID CONSIGLIATO (cr)",
                "bonus_rigorista": "Rigore Bonus"
            }).head(100),
            use_container_width=True
        )
        
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
            df_s = train_prediction_model(INPUT_TEMP, strategy=strategy)

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
            df_all = train_prediction_model(INPUT_TEMP, strategy=strategy)
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
                df_vip = train_prediction_model(INPUT_TEMP, strategy=strategy)

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
        st.markdown("#### Aggiungi override tattico")
        vip_nome    = st.text_input("Nome giocatore", key="vip_add_nome", placeholder="es. Nico Paz")
        vip_pos     = st.selectbox("Posizione tattica", [
            "QUINTO_ATT", "QUINTO_DUTTILE", "ALA", "ALA_TREQUARTISTA",
            "TREQUARTISTA", "MEZZALA_ATT", "MEZZALA_DEF", "REGISTA",
        ], key="vip_pos")
        vip_bonus   = st.slider("Bonus moltiplicatore", 1.05, 1.50, 1.25, 0.01, key="vip_bonus",
                                help="1.0=nessun bonus, 1.5=+50% bonus atteso")
        vip_note    = st.text_input("Note (opzionale)", key="vip_note")
        if st.button("💾 Salva override tattico"):
            try:
                from vip import add_tactical_override
                ok = add_tactical_override(vip_nome, vip_pos, vip_bonus, vip_note)
                if ok:
                    st.success(f"✅ {vip_nome} salvato come {vip_pos} ({vip_bonus:.2f}×)")
                else:
                    st.error("Errore salvataggio")
            except Exception as e:
                st.error(f"Errore: {e}")

        st.markdown("#### Aggiungi giovane promessa")
        vip_nome_y  = st.text_input("Nome giocatore", key="vip_youth_nome", placeholder="es. Camarda")
        vip_anno    = st.number_input("Anno di nascita", 2000, 2010, 2004, 1, key="vip_anno")
        vip_trend   = st.slider("Trend minutaggio", 0.5, 2.0, 1.3, 0.05, key="vip_trend",
                                help="1.0=stabile, >1.0=crescente, <1.0=calante")
        vip_note_y  = st.text_input("Note (opzionale)", key="vip_note_youth")
        if st.button("💾 Salva giovane promessa"):
            try:
                from vip import add_youth_player
                ok = add_youth_player(vip_nome_y, int(vip_anno), vip_trend, vip_note_y)
                if ok:
                    st.success(f"✅ {vip_nome_y} ({CURRENT_YEAR - int(vip_anno)} anni, trend {vip_trend:.2f}) salvato")
                else:
                    st.error("Errore salvataggio")
            except Exception as e:
                st.error(f"Errore: {e}")

        # Riepilogo config attuale
        st.markdown("---")
        st.markdown("#### Config attuale")
        try:
            from vip import get_vip_config_summary
            summary = get_vip_config_summary()
            st.caption(
                f"Versione: {summary['version']} · "
                f"{summary['n_tactical']} override tattici · "
                f"{summary['n_youth']} giovani promesse"
            )
        except Exception:
            st.caption("Config non disponibile")

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
if uploaded_file is not None:
    st.info(f"✨ Dati caricati: **{uploaded_file.name}**")
