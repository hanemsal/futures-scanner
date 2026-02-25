import json
import os
from typing import Any, Dict

class Storage:
    def __init__(self, path: str = "state.json"):
        self.path = path
        self.state: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.state = json.load(f)
            except Exception:
                self.state = {}
        else:
            self.state = {}

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[WARN] state save failed:", e)

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def set(self, key: str, value) -> None:
        self.state[key] = value
