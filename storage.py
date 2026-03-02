import json
import os
import time
from typing import Dict

class Storage:
    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            else:
                self.state = {}
        except Exception:
            self.state = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def mark_sent(self, symbol: str) -> None:
        self.state[symbol] = time.time()
        self._save()

    def is_on_cooldown(self, symbol: str, cooldown_sec: int) -> bool:
        ts = self.state.get(symbol)
        if not ts:
            return False
        return (time.time() - ts) < cooldown_sec
