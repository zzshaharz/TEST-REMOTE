"""Main agent loop — connects screen capture and computer control via Claude API."""

import os
import sys

import anthropic

from screen_capture import ScreenCapture
from computer_control import ComputerControl

MODEL = "claude-sonnet-4-20250514"
MAX_ITERATIONS = 50


def build_tool_definition(image_width: int, image_height: int) -> dict:
    """Build the computer tool definition for the API call."""
    return {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": image_width,
        "display_height_px": image_height,
        "display_number": 1,
    }


def run(task: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    capture = ScreenCapture(max_width=1280, max_height=800)
    screen_w, screen_h = capture.screen_size

    # Grab an initial screenshot to know actual image dimensions
    initial = capture.grab()
    img_w, img_h = initial.size

    control = ComputerControl(screen_w, screen_h, img_w, img_h)
    tool_def = build_tool_definition(img_w, img_h)

    print(f"Screen: {screen_w}x{screen_h} | Image: {img_w}x{img_h}")
    print(f"Task: {task}")
    print("-" * 60)

    # Start with a screenshot so Claude can see the current state
    screenshot_b64 = capture.grab_base64()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": task},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": screenshot_b64,
                    },
                },
            ],
        }
    ]

    system_prompt = (
        "You are controlling a computer via a remote desktop tool. "
        "You can see the screen and execute mouse/keyboard actions. "
        "Use the 'computer' tool to interact with the desktop. "
        "Take screenshots when you need to see the current state. "
        "Complete the user's task step by step."
    )

    for iteration in range(MAX_ITERATIONS):
        print(f"\n[Iteration {iteration + 1}]")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=[tool_def],
            messages=messages,
        )

        # Collect assistant content blocks
        assistant_content = response.content

        # Print any text blocks
        for block in assistant_content:
            if hasattr(block, "text"):
                print(f"Claude: {block.text}")

        # Check if Claude is done (no tool use)
        tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
        if not tool_use_blocks:
            print("\n[Done — Claude finished the task]")
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": assistant_content})

        # Process each tool call and build results
        tool_results = []
        for tool_block in tool_use_blocks:
            action = tool_block.input
            print(f"  Action: {action.get('action', '?')} | {action}")

            result_text = control.execute(action)
            print(f"  Result: {result_text}")

            # After every action, take a fresh screenshot
            screenshot_b64 = capture.grab_base64()

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": [
                    {"type": "text", "text": result_text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": screenshot_b64,
                        },
                    },
                ],
            })

        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            print("\n[Done — end_turn]")
            break
    else:
        print("\n[Stopped — max iterations reached]")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py 'open notepad and type hello world'")
        sys.exit(1)

    run(" ".join(sys.argv[1:]))
