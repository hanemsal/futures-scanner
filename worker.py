import os
import time
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler

from diplist import build_diplist, save_diplist, load_diplist, render_text
from telegram_bot import send_message, get_updates

TZ = ZoneInfo("Europe/Istanbul")

STORAGE_PATH = os.getenv("STORAGE_PATH", "/tmp/diplist.json")
TOP_N = int(os.getenv("DIPLIST_TOP_N", "40"))

# aynı anda 2 scan başlamasın diye lock
_scan_lock = threading.Lock()
_last_scan_started = 0.0


def public_base_url() -> str:
    # Render domain üretimi (site yok derdin burada çözülüyor)
    svc = os.getenv("RENDER_SERVICE_NAME", "").strip()
    if svc:
        return f"https://{svc}.onrender.com"
    host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    if host:
        if host.startswith("http"):
            return host
        return f"https://{host}"
    return ""


def run_diplist(manual: bool) -> None:
    global _last_scan_started

    # 1) aynı anda ikinci scan’i engelle
    if not _scan_lock.acquire(blocking=False):
        send_message("⏳ DipList zaten çalışıyor. Lütfen bitmesini bekle.")
        return

    try:
        _last_scan_started = time.time()
        send_message(f"⏳ DipList başlıyor ({'MANUEL' if manual else 'SCHEDULE'})...")

        results, meta = build_diplist()
        save_diplist(STORAGE_PATH, results, meta)

        msg = render_text(results, meta, top_n=TOP_N)

        base = public_base_url()
        if base:
            msg += f"\n\nİncele:\n{base}/diplist"
        else:
            msg += "\n\nİncele:\n/diplist (Render domain yoksa sadece bot çıktısı kullan)"

        send_message(msg)

    except Exception as e:
        send_message(f"⚠️ Worker hata: {type(e).__name__}: {e}")
    finally:
        _scan_lock.release()


def diplist_now():
    run_diplist(manual=True)


def setup_scheduler():
    sched = BackgroundScheduler(timezone=TZ)

    # 02:00 / 14:00 / 20:00 TR
    sched.add_job(lambda: run_diplist(manual=False), "cron", hour=2, minute=0, id="dip_02")
    sched.add_job(lambda: run_diplist(manual=False), "cron", hour=14, minute=0, id="dip_14")
    sched.add_job(lambda: run_diplist(manual=False), "cron", hour=20, minute=0, id="dip_20")

    sched.start()
    send_message("⏰ Schedule aktif (Europe/Istanbul) -> 02:00,14:00,20:00")


def telegram_loop():
    offset = None
    send_message("🤖 Worker başladı. Komutlar: /diplist | /diplist now")

    while True:
        try:
            data = get_updates(offset=offset, timeout_sec=30)
            if not data.get("ok"):
                time.sleep(2)
                continue

            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                msg = (upd.get("message") or {}).get("text") or ""
                msg = msg.strip()

                if msg == "/diplist":
                    payload = load_diplist(STORAGE_PATH)
                    if not payload:
                        send_message("DipList bulunamadı. Önce /diplist now ile üret.")
                        continue
                    results = payload.get("results", [])
                    meta = payload.get("meta", {})
                    out = render_text(results, meta, top_n=TOP_N)
                    base = public_base_url()
                    if base:
                        out += f"\n\nİncele:\n{base}/diplist"
                    send_message(out)

                elif msg in ["/diplist now", "/diplist_now"]:
                    diplist_now()

        except Exception as e:
            send_message(f"⚠️ Worker hata: {type(e).__name__}: {e}")
            time.sleep(3)


def main():
    setup_scheduler()
    telegram_loop()


if __name__ == "__main__":
    main()
