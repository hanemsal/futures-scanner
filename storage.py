# storage.py
import json
import os
import time
from typing import Any, Dict, Optional


class Storage:
    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Any] = {"sent": {}}  # key -> last_ts
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            if "sent" not in self.data:
                self.data["sent"] = {}
        except Exception:
            # bozuksa sıfırla
            self.data = {"sent": {}}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get_last_sent(self, key: str) -> Optional[float]:
        v = self.data.get("sent", {}).get(key)
        return float(v) if v is not None else None

    def mark_sent(self, key: str) -> None:
        self.data["sent"][key] = time.time()
        self._save()
