"""
news_agent.py — Agente AI Analizzatore Notizie Tattiche e Infortuni.

Utilizza regole NLP avanzate e/o Modelli Linguistici (LLM Gemini / OpenAI)
per estrarre dalle news quotidiane su Serie A e Calciomercato:
- Variazioni di titolarità
- Infortuni e tempi di recupero
- Designazione rigoristi e punizionisti
- Ballottaggi d'asta
- Impatto sui punteggi attesi (previsione_ia)
"""

import os
import json
import logging
import pandas as pd
from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# Keywords estese con pesi tattici ponderati
TACTICAL_KEYWORDS = {
    'infortunio muscolare': 0.75,
    'lesione': 0.65,
    'operazione': 0.40,
    'lungo degenza': 0.40,
    'stop 1 mese': 0.60,
    'stop 2 mesi': 0.45,
    'stop 3 mesi': 0.30,
    'infortunio': 0.80,
    'problema fisico': 0.85,
    'affaticamento': 0.90,
    'recuperato': 1.10,
    'rientro': 1.08,
    'rigorista': 1.15,
    'primo rigorista': 1.18,
    'punizionista': 1.06,
    'titolare inamovibile': 1.15,
    'titolare': 1.08,
    'ballottaggio': 0.92,
    'panchina': 0.82,
    'seconda scelta': 0.75,
    'ceduto': 0.00,
    'fuori rosa': 0.10,
    'acquisto ufficiale': 1.10,
    'forma straordinaria': 1.12,
}


def _match_player_in_text(text: str, player_names: list[str]) -> list[tuple[str, float]]:
    """
    Associa le righe di testo ai giocatori registrati nel database.
    """
    matches = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in lines:
        line_lower = line.lower()
        for name in player_names:
            name_clean = str(name).strip().lower()
            surname = name_clean.split()[-1] if ' ' in name_clean else name_clean

            # Direct string containment or last name match
            if name_clean in line_lower or (len(surname) > 3 and surname in line_lower):
                # Calcola il fattore impatto dalle parole chiave
                impact = 1.0
                matched_kw = []
                for kw, mod in TACTICAL_KEYWORDS.items():
                    if kw in line_lower:
                        impact *= mod
                        matched_kw.append(kw)
                if impact != 1.0:
                    matches.append((name, impact, matched_kw, line))
                    
    return matches


def analyze_news_with_llm(news_text: str, player_names: list[str]) -> dict[str, float]:
    """
    Se è configurata una API key Gemini / OpenAI, interroga l'LLM per estrarre la tabella
    impatto giocatori. Altrimenti passa al fallback NLP.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not gemini_key and not openai_key:
        return {}

    prompt = f"""
Sei un esperto analista di Fantacalcio. Analizza il seguente testo di notizie e trasferimenti:
---
{news_text}
---
Per ogni giocatore menzionato, determina il modificatore della sua prestazione/titolarità attesa (da 0.20 per infortunio grave a 1.25 per nuovo rigorista/titolare).
Restituisci ESCLUSIVAMENTE un oggetto JSON dove le chiavi sono i nomi dei giocatori e i valori sono i moltiplicatori (float).
Esempio: {{"Lautaro Martinez": 0.8, "Vlahovic": 1.15}}
"""
    try:
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content(prompt)
            text_resp = res.text.strip()
            # Pulizia markdown JSON
            if "```" in text_resp:
                text_resp = text_resp.split("```")[1].replace("json", "").strip()
            return json.loads(text_resp)
    except Exception as e:
        logger.warning(f"Errore chiamata LLM News Agent: {e}")
        
    return {}


def apply_news_modifiers(df: pd.DataFrame, news_text: str) -> pd.DataFrame:
    """
    Applica i modificatori estratti dal testo delle news alla colonna 'previsione_ia'.
    """
    if not news_text or not news_text.strip():
        return df

    df = df.copy()
    if 'previsione_ia' not in df.columns:
        df['previsione_ia'] = df.get('fanta_media', 6.0)

    player_names = df['nome'].tolist()
    
    # 1. Tentativo LLM se configurato
    llm_mods = analyze_news_with_llm(news_text, player_names)
    if llm_mods:
        logger.info("Estratti modificatori news via LLM.")
        for name, mod in llm_mods.items():
            mask = df['nome'].str.lower() == name.lower()
            if mask.any():
                df.loc[mask, 'previsione_ia'] *= float(mod)
        return df

    # 2. Fallback NLP avanzato con regole
    matches = _match_player_in_text(news_text, player_names)
    if matches:
        for name, impact, kws, line in matches:
            mask = df['nome'] == name
            df.loc[mask, 'previsione_ia'] *= impact
            logger.info(f"News Agent [NLP]: {name} -> modificatore {impact:.2f} (parole: {kws})")

    df['previsione_ia'] = df['previsione_ia'].round(2)
    return df
