"""Voice notification at the end of a run (bash EXEC_user_notification_WIN_T2S, parity F13).

Windows: SAPI.SpVoice through pywin32 COM (no powershell). macOS: `say`. Linux: `spd-say` or `espeak`,
else `notify-send` (a desktop popup). Off with `--no-notify`, `[defaults].notify = false` or
MSE_LLM_NO_NOTIFY=1. Never raises: a missing speech engine only returns False.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from mse_llm.util import IS_WINDOWS


def disabled() -> bool:
    return os.environ.get("MSE_LLM_NO_NOTIFY", "").strip().lower() in ("1", "true", "yes", "on")


def speak(text: str, timeout: float = 30.0) -> bool:
    try:
        if IS_WINDOWS:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            try:
                win32com.client.Dispatch("SAPI.SpVoice").Speak(text)
            finally:
                pythoncom.CoUninitialize()
            return True
        if sys.platform == "darwin" and shutil.which("say"):
            subprocess.run(["say", text], timeout=timeout, check=False)
            return True
        for tool in (["spd-say", "--wait"], ["espeak"]):
            if shutil.which(tool[0]):
                subprocess.run([*tool, text], timeout=timeout, check=False)
                return True
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", "mse-llm", text], timeout=timeout, check=False)
            return True
    except Exception:
        return False
    return False


def notify(text: str, enabled: bool = True) -> bool:
    """Speak `text` unless notifications are off. Returns whether something was said."""
    if not enabled or disabled() or not text:
        return False
    return speak(text)
