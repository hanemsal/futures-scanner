import json
import os
import time
from typing import Any, Dict, Optional


class Storage:
    """
    Basit JSON dosya storage.
    - cooldown_sec: aynı key'in tekrar sinyal vermesini engeller
    """

    def __init__(self, path: str, enabled: bool = True, cooldown_sec: int = 3600):
        self.path = (path or "state.json").strip()
        self.enabled = bool(enabled)
        self.cooldown_sec = int(cooldown_sec) if cooldown_sec is not None else 3600
        self._data: Dict[str, Any] = {"sent": {}}

        if self.enabled:
            self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {"sent": {}}
            if "sent" not in self._data or not isinstance(self._data["sent"], dict):
                self._data["sent"] = {}
        except Exception:
            self._data = {"sent": {}}

    def _save(self) -> None:
        if not self.enabled:
            return
        try:
            dir_name = os.path.dirname(self.path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def should_send(self, key: str, now_ts: Optional[int] = None) -> bool:
        """cooldown dolduysa True"""
        if not self.enabled:
            return True

        now = int(now_ts or time.time())
        last = self._data.get("sent", {}).get(key)

        if last is None:
            return True

        try:
            last_i = int(last)
        except Exception:
            return True

        return (now - last_i) >= self.cooldown_sec

    def mark_sent(self, key: str, now_ts: Optional[int] = None) -> None:
        if not self.enabled:
            return
        now = int(now_ts or time.time())
        self._data.setdefault("sent", {})[key] = now
        self._save()
