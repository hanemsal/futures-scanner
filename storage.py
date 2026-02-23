import json
import os
import time
from typing import Any, Dict, Optional

DEFAULT_PATH = os.getenv("STORAGE_PATH", "state.json")


def _now() -> int:
    return int(time.time())


def _load(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"symbols": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"symbols": {}}


def _save(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class Storage:
    """
    Per symbol:
      - last_1h_close_time: en son işlenen 1H mumun closeTime'ı (ms)
    """
    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        self.data = _load(self.path)

    def get_symbol_state(self, symbol: str) -> Dict[str, Any]:
        return self.data.setdefault("symbols", {}).setdefault(symbol, {})

    def get_last_close_time(self, symbol: str) -> int:
        st = self.get_symbol_state(symbol)
        return int(st.get("last_1h_close_time", 0) or 0)

    def set_last_close_time(self, symbol: str, close_time_ms: int) -> None:
        st = self.get_symbol_state(symbol)
        st["last_1h_close_time"] = int(close_time_ms)
        _save(self.path, self.data)
