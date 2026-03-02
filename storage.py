import json
import os
import time
from typing import Any, Dict


class Storage:
    def __init__(self, path: str):
        self.path = path
        self.state: Dict[str, Any] = {"cooldowns": {}}
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            if "cooldowns" not in self.state:
                self.state["cooldowns"] = {}
        except Exception as e:
            print(f"[WARN] Storage load failed: {e}")
            self.state = {"cooldowns": {}}

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[WARN] Storage save failed: {e}")

    def can_send(self, key: str, cooldown_sec: int) -> bool:
        now = int(time.time())
        last = int(self.state["cooldowns"].get(key, 0))
        return (now - last) >= cooldown_sec

    def mark_sent(self, key: str) -> None:
        self.state["cooldowns"][key] = int(time.time())
        self.save()
