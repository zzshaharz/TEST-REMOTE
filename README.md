# Claude Remote Desktop Agent

A local application that gives Claude vision and control over your computer — like a remote desktop, but the "remote user" is Claude.

## How it works

```
┌─────────────┐     screenshot     ┌───────────┐
│  Your PC    │ ──────────────────►│  Claude   │
│  (Python)   │                    │  API      │
│             │ ◄──────────────────│           │
│  pyautogui  │   mouse/keyboard   │  Vision + │
│  mss        │   commands          │  Reasoning│
└─────────────┘                    └───────────┘
```

**Loop:**
1. Capture screen → send to Claude as image
2. Claude sees the screen, decides what to do
3. Claude sends back an action (click, type, scroll…)
4. App executes the action via pyautogui
5. Capture new screen → repeat

## Setup

```bash
# 1. Clone and install
git clone https://github.com/zzshaharz/test-remote.git
cd test-remote
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run
python agent.py "open the browser and search for weather"
```

## Requirements

- Python 3.10+
- A display (X11/Wayland on Linux, native on macOS/Windows)
- Anthropic API key with access to Claude's computer-use capability

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Main loop — orchestrates capture → API → execute |
| `screen_capture.py` | Grabs screenshots using `mss`, returns base64 |
| `computer_control.py` | Executes mouse/keyboard via `pyautogui` |

## Supported Actions

- `left_click`, `right_click`, `double_click` — mouse clicks at coordinates
- `mouse_move` — move cursor
- `left_click_drag` — drag and drop
- `type` — type text
- `key` — keyboard shortcuts (e.g., `ctrl+c`)
- `scroll` — scroll up/down
- `screenshot` — request a fresh screenshot
- `wait` — pause execution

## Safety

- `pyautogui.FAILSAFE = True` — move mouse to top-left corner to abort
- Max 50 iterations per run
- All actions are logged to the terminal
