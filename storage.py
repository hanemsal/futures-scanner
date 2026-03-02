import json
import os
import time
from typing import Any, Dict

class Storage:
    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._data: Dict[str, Any] = {}
        self._loaded = False

        # ensure dir exists
        if self.enabled:
            d = os.path.dirname(self.path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

    def _load(self):
        if not self.enabled or self._loaded:
            return
        self._loaded = True
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {}
            else:
                self._data = {}
        except Exception:
            self._data = {}

    def _save(self):
        if not self.enabled:
            return
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def get_int(self, key: str, default: int = 0) -> int:
        self._load()
        try:
            return int(self._data.get(key, default))
        except Exception:
            return default

    def set_int(self, key: str, value: int):
        self._load()
        self._data[key] = int(value)
        self._save()
