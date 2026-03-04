# storage.py
import json
import os
import time
from typing import Dict, Any


class Storage:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {"sent": {}}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            if "sent" not in self.data:
                self.data["sent"] = {}
        except Exception:
            self.data = {"sent": {}}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _key(self, symbol: str, side: str) -> str:
        return f"{symbol}:{side}"

    def is_in_cooldown(self, symbol: str, side: str, cooldown_sec: int) -> bool:
        if cooldown_sec <= 0:
            return False
        key = self._key(symbol, side)
        last = self.data["sent"].get(key)
        if not last:
            return False
        return (time.time() - float(last)) < cooldown_sec

    def mark_sent(self, symbol: str, side: str) -> None:
        key = self._key(symbol, side)
        self.data["sent"][key] = time.time()
        self._save()
