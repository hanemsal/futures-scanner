import json
import os
import time
from typing import Any, Dict, Optional


class Storage:
    """
    Basit JSON state:
    {
      "symbols": {
        "XRPUSDT": {"entry": 1710000000},
        "XRPUSDT:exit": {"exit": 1710000500}
      }
    }
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.state: Dict[str, Any] = {"symbols": {}}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self._save()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            if "symbols" not in self.state:
                self.state = {"symbols": {}}
        except Exception:
            self.state = {"symbols": {}}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get_last(self, key: str, event: str) -> Optional[int]:
        sym = self.state.get("symbols", {}).get(key, {})
        v = sym.get(event)
        return int(v) if v else None

    def mark_event(self, key: str, event: str):
        now = int(time.time())
        if "symbols" not in self.state:
            self.state["symbols"] = {}
        if key not in self.state["symbols"]:
            self.state["symbols"][key] = {}
        self.state["symbols"][key][event] = now
        self._save()

    def in_cooldown(self, key: str, cooldown_sec: int) -> bool:
        last = self.get_last(key, "entry") or self.get_last(key, "exit")
        if not last:
            return False
        return (int(time.time()) - int(last)) < int(cooldown_sec)
