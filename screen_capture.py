"""Screen capture module — grabs the screen and returns base64-encoded images."""

import base64
import io

import mss
from PIL import Image


class ScreenCapture:
    """Captures the primary monitor and returns resized screenshots."""

    def __init__(self, max_width: int = 1280, max_height: int = 800):
        self.max_width = max_width
        self.max_height = max_height
        self.sct = mss.mss()

    @property
    def monitor(self) -> dict:
        return self.sct.monitors[1]  # primary monitor

    @property
    def screen_size(self) -> tuple[int, int]:
        mon = self.monitor
        return mon["width"], mon["height"]

    def grab(self) -> Image.Image:
        """Grab a screenshot and resize it to fit within max dimensions."""
        raw = self.sct.grab(self.monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        # Scale down if needed while preserving aspect ratio
        img.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
        return img

    def grab_base64(self) -> str:
        """Return a screenshot as a base64-encoded JPEG string."""
        img = self.grab()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")
