import json
import os
import time
from typing import Dict, Any

class Storage:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path:
            self.data = {}
            return
        if not os.path.exists(self.path):
            self.data = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {}

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def can_send(self, key: str, cooldown_sec: int) -> bool:
        now = int(time.time())
        last = int(self.data.get(key, 0) or 0)
        return (now - last) >= cooldown_sec

    def mark_sent(self, key: str) -> None:
        self.data[key] = int(time.time())
        self.save()
