import os
import time
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.background import BackgroundScheduler

from diplist import build_diplist, save_diplist, load_diplist
from telegram_bot import send_message, get_updates, is_allowed_chat, extract_text

STORAGE_PATH = os.getenv("STORAGE_PATH", "/var/data/diplist.json")

ENABLE_SCHEDULE = os.getenv("ENABLE_SCHEDULE", "0") == "1"
SCHEDULE_TIMES = os.getenv("DIPLIST_SCHEDULE", "02:00,14:00,20:00")
TZ_NAME = os.getenv("TZ_NAME", "Europe/Istanbul")

TOP_N = int(os.getenv("DIPLIST_TOP_N", "40"))


def format_items(payload, top_n: int = TOP_N) -> str:
    if not payload:
        return "DipList bulunamadı. Önce /diplist now ile üret."

    meta = payload.get("meta", {})
    items = payload.get("items", [])
    gen = payload.get("generated_at_utc", "?")

    lines = []
    lines.append(f"📌 DipList (UTC): {gen}")
    lines.append(
        f"🔎 Scanned: {meta.get('symbols_scanned')} | Dip: {meta.get('dip_count')} | Err: {meta.get('errors')} | t={meta.get('elapsed_sec')}s"
    )
    lines.append("")

    for it in items[:top_n]:
        sym = it.get("tv_symbol") or it.get("symbol")
        r1m = it.get("rsi_1m")
        r1w = it.get("rsi_1w")
        r1d = it.get("rsi_1d")
        r4h = it.get("rsi_4h")
        r1h = it.get("rsi_1h")
        reasons = ",".join(it.get("reasons", []))[:180]
        lines.append(f"{sym} | 1M:{r1m} 1W:{r1w} 1D:{r1d} 4H:{r4h} 1H:{r1h} | {reasons}")

    if len(items) > top_n:
        lines.append("")
        lines.append(f"… toplam {len(items)} coin. (Top {top_n} gösterildi)")
    return "\n".join(lines)


def run_diplist_and_store(manual: bool = False) -> None:
    tag = "MANUEL" if manual else "SCHEDULE"
    send_message(f"⏳ DipList başlıyor ({tag})…")

    items, meta = build_diplist()
    save_diplist(STORAGE_PATH, items, meta)

    payload = load_diplist(STORAGE_PATH)
    send_message("✅ DipList hazır.\n\n" + format_items(payload))


def setup_scheduler() -> BackgroundScheduler:
    tz = ZoneInfo(TZ_NAME)
    sched = BackgroundScheduler(timezone=tz)

    for t in [x.strip() for x in SCHEDULE_TIMES.split(",") if x.strip()]:
        hh, mm = t.split(":")
        sched.add_job(
            func=run_diplist_and_store,
            trigger="cron",
            hour=int(hh),
            minute=int(mm),
            id=f"diplist_{hh}{mm}",
            replace_existing=True,
        )

    sched.start()
    return sched


def main():
    send_message("🤖 Worker başladı. Komutlar: /diplist  |  /diplist now")

    if ENABLE_SCHEDULE:
        setup_scheduler()
        send_message(f"⏰ Schedule aktif ({TZ_NAME}) -> {SCHEDULE_TIMES}")

    offset = None

    while True:
        try:
            data = get_updates(offset=offset, timeout_sec=25)

            if not data.get("ok"):
                time.sleep(1)
                continue

            for upd in data.get("result", []):
                offset = upd["update_id"] + 1

                if not is_allowed_chat(upd):
                    continue

                text = extract_text(upd)
                if not text:
                    continue

                cmd = text.strip()

                # Sistemden bağımsız: son kaydı göster
                if cmd == "/diplist":
                    payload = load_diplist(STORAGE_PATH)
                    send_message(format_items(payload))

                # Sistemden bağımsız: anında üret
                elif cmd.startswith("/diplist"):
                    parts = cmd.split()
                    if len(parts) >= 2 and parts[1].lower() == "now":
                        run_diplist_and_store(manual=True)
                    else:
                        send_message("Kullanım: /diplist  veya  /diplist now")

        except requests.exceptions.ReadTimeout:
            # long polling: normal
            continue
        except Exception as e:
            send_message(f"⚠️ Worker hata: {type(e).__name__}")
            time.sleep(2)


if __name__ == "__main__":
    main()
