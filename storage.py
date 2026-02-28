import json
import os
import time
from typing import Any, Dict, Optional

class Storage:
    def __init__(self, path: str):
        self.path = path
        self._cache: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f) or {}
            else:
                self._cache = {}
        except Exception:
            # dosya bozuksa sıfırdan başla
            self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        self._ensure_loaded()
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._ensure_loaded()
        self._cache[key] = value
        self._persist()

    def _persist(self) -> None:
        # klasör yoksa oluştur
        d = os.path.dirname(self.path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
