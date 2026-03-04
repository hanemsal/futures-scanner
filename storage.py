# storage.py
import json
import os
import time
from typing import Any, Dict, Optional


class Storage:
    """
    Basit JSON storage (atomic write).
    Render disk mount: /var/data kullanacaksın.
    """
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
            # bozuk dosya vs. olursa sıfırdan başlar
            self._cache = {}

    def get(self, key: str, default: Any = None) -> Any:
        self._ensure_loaded()
        return self._cache.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._ensure_loaded()
        self._cache[key] = value
        self._flush()

    def _flush(self) -> None:
        # Atomic write: önce tmp yaz, sonra rename
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp.{int(time.time()*1000)}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)
