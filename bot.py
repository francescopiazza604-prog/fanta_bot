import os
import logging
import tempfile

import pandas as pd
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
ALLOWED_USER_ID = int(os.environ.get("TELEGRAM_USER_ID", "0"))

# Cache in memoria per le 5 squadre generate (per sessione utente)
_last_multi_teams: dict[int, list[dict]] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
QUOTAZIONI_PATH = os.path.join(DATA_DIR, "quotazioni_correnti.csv")
FALLBACK_PATH = os.path.join(DATA_DIR, "serie_a_23_24_backtest.csv")


# ── helpers ──────────────────────────────────────────────────────────────────

def _auth(update: Update) -> bool:
    if ALLOWED_USER_ID == 0:
        return True
    return update.effective_user.id == ALLOWED_USER_ID


def _active_data_path() -> str | None:
    if os.path.exists(QUOTAZIONI_PATH):
        return QUOTAZIONI_PATH
    if os.path.exists(FALLBACK_PATH):
        return FALLBACK_PATH
    return None


def _run_predictions() -> pd.DataFrame | None:
    path = _active_data_path()
    if not path:
        return None
    from predict import train_prediction_model
    return train_prediction_model(path)


# ── commands ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return
    await update.message.reply_text(
        "🤖 Ciao! Sono FantaBot, il tuo assistente IA per il Fantacalcio.\n\n"
        "Comandi principali:\n"
        "📊 /top — Migliori scommesse qualità/prezzo\n"
        "🏆 /ottimizza [budget] — 5 squadre a confronto (default: 500)\n"
        "🔍 /giocatore <nome> — Analisi giocatore\n"
        "📈 /forma <nome> <v1> <v2> [v3] [v4] — Aggiorna ultimi voti\n\n"
        "Modello IA:\n"
        "🧠 /allena [stagione] — Addestra il modello temporale (migliore!)\n"
        "📋 /modello — Stato del modello attivo\n\n"
        "Dati:\n"
        "🔄 /aggiorna — Aggiorna stats FBref\n"
        "📤 /carica — Come caricare le quotazioni\n"
        "📰 /news — Ultime notizie\n\n"
        "Puoi anche inviarmi direttamente il file .csv o .xlsx delle quotazioni!\n\n"
        "💡 Tip: carica le quotazioni Fantacalcio.it (con FM), poi usa /allena per "
        "attivare il modello temporale — previsioni causalmente corrette."
    )


async def cmd_top(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return
    msg = await update.message.reply_text("🔄 Calcolo previsioni in corso...")

    try:
        df = _run_predictions()
        if df is None:
            await msg.edit_text("❌ Nessun dato disponibile. Usa /carica per caricare le quotazioni.")
            return

        df['convenienza'] = df['previsione_ia'] / (df['costo_iniziale'].clip(lower=1) + 2)
        
        # Seleziona top per ogni reparto (2P, 3D, 3C, 3A) per evitare monopolio dei portieri
        top_list = []
        for r_code, count in [('P', 2), ('D', 3), ('C', 3), ('A', 3)]:
            sub = df[df['ruolo'] == r_code].nlargest(count, 'convenienza')
            top_list.append(sub)
        top = pd.concat(top_list)

        model_badge = "🧠 modello temporale" if df.get('_temporal_model', pd.Series([False])).any() else "⚠️ modello same-season"
        lines = [f"🏅 *TOP SCOMMESSE PER REPARTO (Qualità/Prezzo)* — {model_badge}\n"]
        for _, row in top.iterrows():
            gol = f"{row['gol_pg']:.2f}" if 'gol_pg' in row and pd.notna(row.get('gol_pg')) else "0.00"
            lines.append(
                f"• *{row['nome']}* [{row['ruolo']}]\n"
                f"  Prev IA: *{row['previsione_ia']:.1f}* | "
                f"Gol/g: {gol} | Costo: *{int(row['costo_iniziale'])} cr.*"
            )

        await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"cmd_top error: {e}")
        await msg.edit_text(f"❌ Errore: {e}")


async def cmd_ottimizza(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return

    budget = 500
    if context.args:
        try:
            budget = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Budget non valido. Esempio: /ottimizza 500")
            return

    msg = await update.message.reply_text(
        f"🔄 Genero 5 squadre con budget {budget} cr. (può richiedere 15–30 sec)..."
    )

    try:
        path = _active_data_path()
        if not path:
            await msg.edit_text("❌ Nessun dato disponibile. Usa /carica per caricare le quotazioni.")
            return

        from optimizer import optimize_team_multi
        teams_data = optimize_team_multi(path, budget=budget)

        if not teams_data:
            await msg.edit_text("❌ Nessuna formazione trovata con questo budget.")
            return

        user_id = update.effective_user.id
        _last_multi_teams[user_id] = teams_data

        # ── Messaggio di confronto ────────────────────────────────────────────
        lines = [f"⚽ *5 STRATEGIE A CONFRONTO* — Budget {budget} cr.\n"]

        emoji_num = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for i, td in enumerate(teams_data):
            sp = td['spesa_per_ruolo']
            lines.append(
                f"{emoji_num[i]} *{td['label']}* — {td['tagline']}\n"
                f"   FM titolari: *{td['punti_previsti']}* | Costo: {td['costo_totale']}/{budget} cr.\n"
                f"   P:{sp['P']} D:{sp['D']} C:{sp['C']} A:{sp['A']}"
            )
            if td['pro']:
                lines.append(f"   ✅ {td['pro'][0]}")
            if td['contro']:
                lines.append(f"   ⚠️ {td['contro'][0]}")
            lines.append("")

        lines.append("👇 Clicca per vedere la rosa completa:")

        # ── Inline keyboard ───────────────────────────────────────────────────
        keyboard = [
            [InlineKeyboardButton(
                f"📋 {td['label']} ({td['punti_previsti']} FM)",
                callback_data=f"team_{i}_{user_id}",
            )]
            for i, td in enumerate(teams_data)
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await msg.edit_text(
            "\n".join(lines),
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )

    except ValueError as e:
        await msg.edit_text(f"⚠️ {e}")
    except Exception as e:
        logger.error(f"cmd_ottimizza error: {e}")
        await msg.edit_text(f"❌ Errore: {e}")


async def callback_team_detail(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Mostra la rosa completa di una delle 5 squadre generate."""
    query = update.callback_query
    await query.answer()

    try:
        parts   = query.data.split("_")   # "team_{idx}_{user_id}"
        idx     = int(parts[1])
        user_id = int(parts[2])
    except (IndexError, ValueError):
        await query.message.reply_text("❌ Dati non validi.")
        return

    teams_data = _last_multi_teams.get(user_id)
    if not teams_data or idx >= len(teams_data):
        await query.message.reply_text(
            "❌ Sessione scaduta — rigenera le squadre con /ottimizza"
        )
        return

    td     = teams_data[idx]
    team   = td['team']
    budget = td['budget']

    lines = [
        f"✅ *{td['label']}* — {td['tagline']}\n",
        f"FM prevista titolari: *{td['punti_previsti']}* | Costo: {td['costo_totale']}/{budget} cr.",
        f"P:{td['spesa_per_ruolo']['P']} D:{td['spesa_per_ruolo']['D']} "
        f"C:{td['spesa_per_ruolo']['C']} A:{td['spesa_per_ruolo']['A']}\n",
    ]

    # Pro/contro
    for item in td['pro'][:3]:
        lines.append(f"✅ {item}")
    for item in td['contro'][:2]:
        lines.append(f"⚠️ {item}")
    lines.append("")

    # Rosa completa per ruolo
    from predict import build_compact_reason
    ruolo_labels = {'P': '🧤 Portieri', 'D': '🛡 Difensori', 'C': '⚙️ Centrocampisti', 'A': '⚽ Attaccanti'}
    for ruolo in ['P', 'D', 'C', 'A']:
        subset = team[team['ruolo'] == ruolo].sort_values('costo_iniziale', ascending=False)
        if subset.empty:
            continue
        lines.append(f"\n*{ruolo_labels[ruolo]}*")

        n_tit = {'P': 1, 'D': 4, 'C': 4, 'A': 3}[ruolo]
        for j, (_, row) in enumerate(subset.iterrows()):
            star   = "★" if j < n_tit else "☆"
            reason = build_compact_reason(row)
            reason_str = f"\n    ↳ _{reason}_" if reason else ""
            cs_str = (
                f" | CS: {row.get('clean_sheet_pg', 0):.0%}"
                if ruolo == 'P' and row.get('clean_sheet_pg', 0) > 0
                else ""
            )
            lines.append(
                f"  {star} {row['nome']} — "
                f"{int(row['costo_iniziale'])} cr. | "
                f"FM: {row.get('previsione_ia', 0):.1f}"
                f"{cs_str}{reason_str}"
            )

    await query.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_giocatore(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return

    if not context.args:
        await update.message.reply_text("Uso: /giocatore <nome>\nEsempio: /giocatore Vlahovic")
        return

    nome_cercato = " ".join(context.args).lower()

    try:
        df = _run_predictions()
        if df is None:
            await update.message.reply_text("❌ Nessun dato disponibile.")
            return

        mask = df['nome'].str.lower().str.contains(nome_cercato, na=False)
        risultati = df[mask].head(3)

        if risultati.empty:
            await update.message.reply_text(f"❌ Nessun giocatore trovato per '{nome_cercato}'.")
            return

        from predict import build_player_explanation
        for _, row in risultati.iterrows():
            testo = build_player_explanation(row)
            await update.message.reply_text(testo, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"cmd_giocatore error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_forma(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /forma <nome giocatore> <voto1> <voto2> [voto3] [voto4]
    Salva gli ultimi N voti del giocatore per la feature forma recente.
    La cache del modello viene invalidata automaticamente.
    """
    if not _auth(update):
        return

    if not context.args or len(context.args) < 3:
        await update.message.reply_text(
            "📈 Uso: /forma <nome> <voto1> <voto2> [voto3] [voto4]\n\n"
            "Esempi:\n"
            "  /forma Vlahovic 8.5 5.5 7.0\n"
            "  /forma Lautaro Martinez 9.0 6.0 7.5 8.0\n\n"
            "I voti aggiornano la feature 'forma recente' del modello IA.\n"
            "La cache viene invalidata automaticamente."
        )
        return

    # Separa il nome dai voti partendo dalla fine (gli ultimi token numerici sono voti)
    args = context.args
    voti: list[float] = []
    i = len(args) - 1
    while i >= 0:
        try:
            voti.insert(0, float(args[i]))
            i -= 1
        except ValueError:
            break
    nome_parts = args[:i + 1]

    if not nome_parts or len(voti) < 2:
        await update.message.reply_text(
            "❌ Formato non valido.\n"
            "Usa: /forma <nome> <voto1> <voto2> [voto3] [voto4]\n"
            "Servono almeno 2 voti numerici."
        )
        return

    if any(v < 1 or v > 15 for v in voti):
        await update.message.reply_text("❌ I voti devono essere tra 1 e 15.")
        return

    nome = " ".join(nome_parts)

    try:
        from predict import save_forma_recente
        ok = save_forma_recente(nome, voti)
        if ok:
            media = round(sum(voti) / len(voti), 2)
            trend = "📈 In forma" if media >= 7.0 else "📉 Fuori forma" if media < 5.5 else "➡️ Nella media"
            await update.message.reply_text(
                f"✅ Forma aggiornata per *{nome}*\n"
                f"Voti: {' | '.join(f'{v:.1f}' for v in voti)}\n"
                f"Media recente: *{media}* — {trend}\n\n"
                f"Il modello userà questa info alla prossima /ottimizza o /top.",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text("❌ Errore nel salvataggio. Riprova.")
    except Exception as e:
        logger.error(f"cmd_forma error: {e}")
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_allena(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /allena [stagione]
    Addestra il modello temporale corretto: usa FBref stats stagione N come features
    e la FM dalla colonna fanta_media del file quotazioni come target.

    Questo produce previsioni causalmente corrette: il modello impara
    cosa succede nella stagione SUCCESSIVA dato le stats di quella PRECEDENTE.
    """
    if not _auth(update):
        return

    feature_season = "2022-2023"
    if context.args:
        feature_season = context.args[0]

    path = _active_data_path()
    if not path:
        await update.message.reply_text(
            "❌ Nessun file quotazioni trovato.\n"
            "Carica prima le quotazioni (con /carica o inviando il file .csv/.xlsx).\n\n"
            "Il file deve avere la colonna *fanta_media* (FM della stagione scorsa).",
            parse_mode="Markdown",
        )
        return

    # Verifica che il file quotazioni abbia fanta_media
    try:
        df_check = pd.read_csv(path)
        has_fm = 'fanta_media' in df_check.columns and df_check['fanta_media'].notna().sum() >= 30
    except Exception:
        has_fm = False

    if not has_fm:
        await update.message.reply_text(
            "⚠️ Il file quotazioni non ha la colonna *fanta_media* (o ha troppi valori vuoti).\n\n"
            "Per addestrare il modello temporale, il file quotazioni deve contenere la "
            "fantamedia reale della stagione scorsa. Solitamente il file Excel ufficiale "
            "di Fantacalcio.it la include.",
            parse_mode="Markdown",
        )
        return

    msg = await update.message.reply_text(
        f"🧠 Addestramento modello temporale in corso...\n"
        f"Features: FBref {feature_season} | Target: FM dal file quotazioni\n\n"
        f"⏳ Download FBref in corso (30–60 sec)..."
    )

    try:
        from temporal_model import run_temporal_training, DEFAULT_FEATURE_SEASON
        ok, result_msg, metrics = run_temporal_training(path, feature_season)
        await msg.edit_text(result_msg)
    except Exception as e:
        logger.error(f"cmd_allena error: {e}", exc_info=True)
        await msg.edit_text(
            f"❌ Errore durante l'addestramento:\n{e}\n\n"
            f"Controlla che FBref sia raggiungibile e riprova."
        )


async def cmd_stato_modello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /modello — Mostra quale modello è attivo (temporale o same-season).
    """
    if not _auth(update):
        return
    try:
        from temporal_model import temporal_models_exist, load_temporal_models, TEMPORAL_MODELS_PATH
        import os
        if temporal_models_exist():
            _, metrics, feat_season = load_temporal_models()
            mtime = os.path.getmtime(TEMPORAL_MODELS_PATH)
            import datetime
            dt = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
            lines = [
                "🧠 *Modello temporale ATTIVO*",
                f"Addestrato il: {dt}",
                f"Features da: {feat_season}",
                "",
                "Accuratezza (MAE cross-val per ruolo):",
            ]
            if metrics:
                for ruolo, m in metrics.items():
                    quality = "🟢" if m['mae_cv'] < 0.5 else "🟡" if m['mae_cv'] < 0.8 else "🔴"
                    lines.append(f"  {quality} {ruolo}: ±{m['mae_cv']:.2f} FM ({m['n']} giocatori)")
            lines += [
                "",
                "Questo modello predice la stagione N+1 basandosi sulle stats N.",
                "È superiore al fallback same-season.",
            ]
        else:
            lines = [
                "⚠️ *Modello same-season* (fallback)",
                "",
                "Il modello temporale non è ancora stato addestrato.",
                "Usa */allena* per attivarlo — ti servirà il file quotazioni con la colonna fanta_media.",
                "",
                "Differenza:",
                "• Same-season: addestra e predice sulla STESSA stagione (circolare)",
                "• Temporale: impara da stagione N, predice stagione N+1 (corretto)",
            ]
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Errore: {e}")


async def cmd_news(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return
    msg = await update.message.reply_text("📰 Recupero ultime news...")
    try:
        from news_scraper import fetch_latest_fanta_news
        news = fetch_latest_fanta_news()
        await msg.edit_text(f"📰 ULTIME NEWS FANTACALCIO:\n\n{news}")
    except Exception as e:
        await msg.edit_text(f"❌ Errore: {e}")


async def cmd_aggiorna(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return
    msg = await update.message.reply_text("🔄 Download stats e trasferimenti in corso (30-60 sec)...")
    try:
        from scraper_stats import fetch_and_save_stats
        from scraper_transfermarkt import fetch_and_save_transfers
        
        ok_stats = fetch_and_save_stats()
        ok_tm, tm_msg = fetch_and_save_transfers(2026)
        
        reply = []
        if ok_stats:
            reply.append("✅ Stats Serie A aggiornate da FBref!")
        else:
            reply.append("⚠️ FBref non raggiungibile. Uso dati in cache.")
            
        if ok_tm:
            reply.append("✅ Trasferimenti aggiornati!")
        else:
            reply.append(f"⚠️ Trasferimenti: {tm_msg}")
            
        await msg.edit_text("\n".join(reply))
    except Exception as e:
        logger.error(f"cmd_aggiorna error: {e}")
        await msg.edit_text(f"❌ Errore: {e}")


async def cmd_carica(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return
    await update.message.reply_text(
        "📤 Come caricare le quotazioni:\n\n"
        "1. Vai su Fantacalcio.it\n"
        "2. Sezione Asta → scarica il file Excel/CSV quotazioni\n"
        "3. Inviamelo qui come documento\n\n"
        "Il bot aggiornerà automaticamente le previsioni!"
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return

    doc = update.message.document
    if not doc:
        return

    ext = os.path.splitext(doc.file_name)[1].lower()
    if ext not in ('.csv', '.xlsx'):
        await update.message.reply_text("❌ Formato non supportato. Invia un file .csv o .xlsx")
        return

    msg = await update.message.reply_text("📥 File ricevuto, elaborazione in corso...")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        upload_path = os.path.join(DATA_DIR, f"quotazioni_upload{ext}")
        await tg_file.download_to_drive(upload_path)

        from importer import import_real_quotazioni
        df = import_real_quotazioni(upload_path)

        if df is not None and not df.empty:
            df.to_csv(QUOTAZIONI_PATH, index=False)
            counts = df['ruolo'].value_counts()
            await msg.edit_text(
                f"✅ {len(df)} giocatori caricati!\n"
                f"P:{counts.get('P',0)} D:{counts.get('D',0)} "
                f"C:{counts.get('C',0)} A:{counts.get('A',0)}\n\n"
                f"Usa /top o /ottimizza per analizzare!"
            )
        else:
            await msg.edit_text(
                "❌ Impossibile leggere il file.\n"
                "Assicurati che contenga colonne: Ruolo (R), Nome, Costo/Quotazione."
            )

    except Exception as e:
        logger.error(f"handle_document error: {e}")
        await msg.edit_text(f"❌ Errore nel processare il file: {e}")


async def cmd_listone(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _auth(update):
        return

    budget = 500
    if context.args:
        try:
            budget = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❌ Budget non valido. Esempio: /listone 500")
            return

    msg = await update.message.reply_text(
        f"🏆 Calcolo la Formazione Listone Master (11 Top + 14 Coperture) per budget {budget} cr..."
    )

    try:
        path = _active_data_path()
        if not path:
            await msg.edit_text("❌ Nessun dato disponibile. Usa /carica per caricare le quotazioni.")
            return

        from listone_optimizer import optimize_listone_auto
        df_pred = _run_predictions()
        if df_pred is None:
            await msg.edit_text("❌ Impossibile generare le predizioni.")
            return

        team, best_form, best_score = optimize_listone_auto(df_pred, budget=budget)

        starters = team[team['is_starter']]
        bench = team[~team['is_starter']]

        lines = [
            f"🏆 *FORMULAZIONE LISTONE MASTER* — Modulo *{best_form}*",
            f"📊 Punti Totali Attesi 11 Titolari (38g): *{best_score:.1f} pti*",
            f"💰 Budget Utilizzato: *{int(team['costo_iniziale'].sum())}/{budget} cr.*\n",
            f"⚡ *11 TITOLARI ({best_form}):*"
        ]

        for _, r in starters.iterrows():
            lines.append(f"  • *{r['nome']}* ({r['ruolo']}, {int(r['costo_iniziale'])}cr) — Pti 38g: {r['pts_38g']:.0f}")

        lines.append(f"\n🛡️ *14 RISERVE / COPERTURE:*")
        for _, r in bench.iterrows():
            lines.append(f"  • {r['nome']} ({r['ruolo']}, {int(r['costo_iniziale'])}cr)")

        await msg.edit_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"cmd_listone error: {e}")
        await msg.edit_text(f"❌ Errore: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN non impostato.\n"
            "Crea un file .env con:\n"
            "  TELEGRAM_BOT_TOKEN=il_tuo_token\n"
            "  TELEGRAM_USER_ID=il_tuo_id"
        )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("listone", cmd_listone))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("ottimizza", cmd_ottimizza))
    app.add_handler(CommandHandler("giocatore", cmd_giocatore))
    app.add_handler(CommandHandler("forma", cmd_forma))
    app.add_handler(CommandHandler("allena", cmd_allena))
    app.add_handler(CommandHandler("modello", cmd_stato_modello))
    app.add_handler(CommandHandler("news", cmd_news))
    app.add_handler(CommandHandler("aggiorna", cmd_aggiorna))
    app.add_handler(CommandHandler("carica", cmd_carica))
    app.add_handler(CallbackQueryHandler(callback_team_detail, pattern=r"^team_\d+_\d+$"))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    logger.info("FantaBot avviato. In ascolto...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
