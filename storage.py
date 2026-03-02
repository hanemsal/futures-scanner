import json
import os
import time
from typing import Any, Dict


class Storage:
    """
    Simple JSON state storage to avoid spamming same symbol.
    Stores last_sent_ts per symbol.
    """

    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, Any] = {"last_sent": {}}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            if "last_sent" not in self.state:
                self.state["last_sent"] = {}
        except Exception:
            # If corrupted, reset safely
            self.state = {"last_sent": {}}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def is_cooldown(self, symbol: str, cooldown_sec: int) -> bool:
        last = float(self.state.get("last_sent", {}).get(symbol, 0))
        return (time.time() - last) < cooldown_sec

    def mark_sent(self, symbol: str) -> None:
        self.state["last_sent"][symbol] = time.time()
        self._save()
