"""Computer control module — executes mouse and keyboard actions."""

import time

import pyautogui

# Safety: allow moving to corners (disable pyautogui failsafe if needed)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


class ComputerControl:
    """Translates Claude tool-use actions into real mouse/keyboard events."""

    def __init__(self, screen_width: int, screen_height: int,
                 image_width: int, image_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image_width = image_width
        self.image_height = image_height

    def _scale(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates from image space to actual screen space."""
        sx = int(x * self.screen_width / self.image_width)
        sy = int(y * self.screen_height / self.image_height)
        return sx, sy

    def execute(self, action: dict) -> str:
        """Execute a single computer-use action and return a status string."""
        action_type = action.get("action")

        if action_type == "mouse_move":
            x, y = self._scale(action["coordinate"][0], action["coordinate"][1])
            pyautogui.moveTo(x, y)
            return f"mouse_move to ({x}, {y})"

        if action_type == "left_click":
            x, y = self._scale(action["coordinate"][0], action["coordinate"][1])
            pyautogui.click(x, y)
            return f"left_click at ({x}, {y})"

        if action_type == "right_click":
            x, y = self._scale(action["coordinate"][0], action["coordinate"][1])
            pyautogui.rightClick(x, y)
            return f"right_click at ({x}, {y})"

        if action_type == "double_click":
            x, y = self._scale(action["coordinate"][0], action["coordinate"][1])
            pyautogui.doubleClick(x, y)
            return f"double_click at ({x}, {y})"

        if action_type == "left_click_drag":
            sx, sy = self._scale(action["startCoordinate"][0], action["startCoordinate"][1])
            ex, ey = self._scale(action["coordinate"][0], action["coordinate"][1])
            pyautogui.moveTo(sx, sy)
            pyautogui.mouseDown()
            pyautogui.moveTo(ex, ey, duration=0.5)
            pyautogui.mouseUp()
            return f"drag from ({sx},{sy}) to ({ex},{ey})"

        if action_type == "type":
            pyautogui.typewrite(action["text"], interval=0.03)
            return f"typed: {action['text'][:40]}..."

        if action_type == "key":
            keys = action["text"]
            pyautogui.hotkey(*keys.split("+"))
            return f"key: {keys}"

        if action_type == "scroll":
            x, y = self._scale(action["coordinate"][0], action["coordinate"][1])
            direction = action["direction"]
            amount = action.get("amount", 3)
            scroll_val = amount if direction == "up" else -amount
            pyautogui.scroll(scroll_val, x, y)
            return f"scroll {direction} at ({x},{y})"

        if action_type == "screenshot":
            return "screenshot_requested"

        if action_type == "wait":
            time.sleep(action.get("duration", 1))
            return "waited"

        return f"unknown action: {action_type}"
