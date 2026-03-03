import json
import os
import time
from typing import Any, Dict, Optional


class Storage:
    """
    Persistent state for:
    - cooldown timestamps (per symbol, per signal kind)
    - open positions (entry sent -> exit watch)
    """

    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, Any] = {
            "last_sent": {},   # key -> ts
            "open": {},        # symbol -> {"ts":..., "kind":...}
        }
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.state.update(data)
        except Exception as e:
            print(f"[STORAGE] load failed: {e}")

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[STORAGE] save failed: {e}")

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def is_cooldown(self, key: str, cooldown_sec: int) -> bool:
        last = int(self.state.get("last_sent", {}).get(key, 0) or 0)
        return (self._now() - last) < cooldown_sec

    def mark_sent(self, key: str) -> None:
        self.state.setdefault("last_sent", {})[key] = self._now()
        self._save()

    def set_open(self, symbol: str, kind: str) -> None:
        self.state.setdefault("open", {})[symbol] = {"ts": self._now(), "kind": kind}
        self._save()

    def clear_open(self, symbol: str) -> None:
        if symbol in self.state.get("open", {}):
            del self.state["open"][symbol]
            self._save()

    def get_open(self) -> Dict[str, Any]:
        return dict(self.state.get("open", {}))
