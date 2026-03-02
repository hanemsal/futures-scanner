import json
import os
import time
from typing import Any, Dict, Optional

class Storage:
    def __init__(self, path: str):
        self.path = path
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _read(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.path):
                return {}
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get_last_signal_ts(self, symbol: str) -> Optional[int]:
        data = self._read()
        v = data.get("last_signal_ts", {}).get(symbol)
        if isinstance(v, (int, float)):
            return int(v)
        return None

    def set_last_signal_ts(self, symbol: str, ts: int) -> None:
        data = self._read()
        if "last_signal_ts" not in data or not isinstance(data["last_signal_ts"], dict):
            data["last_signal_ts"] = {}
        data["last_signal_ts"][symbol] = int(ts)
        data["updated_at"] = int(time.time())
        self._write(data)
