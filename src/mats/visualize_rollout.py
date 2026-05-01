"""Render agent rollouts (messages.json) as readable HTML pages.

Resolves one ``messages.json`` (direct path, or last ``step-*`` under a ``run-*``
directory). If you pass a timestamp directory that contains ``run-*`` subfolders,
each run is rendered to its own ``rollout.html`` (no combined index; for a coding
experiment index with test columns use ``coding/visualize_rollout.py``).

Usage:
    uv run python -m mats.visualize_rollout results/.../run-15
    uv run python -m mats.visualize_rollout results/.../step-7/messages.json
    uv run python -m mats.visualize_rollout results/.../2026-03-02_04-02-13
    uv run python -m mats.visualize_rollout .../run-15 --output custom.html
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import fire


def resolve_single_rollout_messages(path: Path) -> tuple[Path, list[dict]]:
    """Resolve to one messages.json path and its contents.

    Accepts:
    - A messages.json file
    - A run directory containing step-* (uses the last step)

    Raises:
        FileNotFoundError: Path is missing or has no usable messages.json.
    """
    if path.is_file():
        if path.name != "messages.json":
            raise FileNotFoundError(f"Not a messages.json file: {path}")
        return path, json.loads(path.read_text())

    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")

    step_dirs = sorted(path.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if not step_dirs:
        raise FileNotFoundError(f"No step-* directories in {path}")

    # Walk backwards to find the last step with a non-empty messages.json
    for step_dir in reversed(step_dirs):
        mj = step_dir / "messages.json"
        if mj.exists() and mj.stat().st_size > 0:
            return mj, json.loads(mj.read_text())
    raise FileNotFoundError(f"No non-empty messages.json in any step of {path}")


def list_rollout_entries(path: Path) -> list[tuple[Path, list[dict]]]:
    """Resolve a path to one or more (messages.json path, messages) pairs.

    - A ``messages.json`` file or a single ``run-*`` directory: one entry.
    - A directory whose immediate children include ``run-*``: one entry per run,
      sorted by run index (timestamp-style layout).
    """
    if path.is_dir():
        run_dirs = sorted(path.glob("run-*"), key=lambda p: int(p.name.split("-")[1]))
        if run_dirs:
            entries = []
            for rd in run_dirs:
                try:
                    entries.append(resolve_single_rollout_messages(rd))
                except FileNotFoundError:
                    print(f"Skipping {rd.name}: no valid messages.json")
            return entries
    return [resolve_single_rollout_messages(path)]


def load_state(messages_json_path: Path) -> dict:
    """Load the state.json that sits next to messages.json."""
    state_path = messages_json_path.parent / "state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    background: #f8f9fa;
    color: #1f2328;
    padding: 24px;
    max-width: 960px;
    margin: 0 auto;
    font-size: 13px;
    line-height: 1.6;
}
h1 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #1f2328;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
    padding-bottom: 12px;
    border-bottom: 1px solid #d1d9e0;
}
.meta {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: #59636e;
    font-size: 13px;
    margin-bottom: 24px;
}
.meta .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-left: 8px;
}
.badge-pass { background: #dafbe1; color: #1a7f37; }
.badge-fail { background: #ffebe9; color: #d1242f; }
.step-divider {
    text-align: center;
    margin: 20px 0;
    color: #8b949e;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.step-divider span {
    background: #f8f9fa;
    padding: 0 12px;
}
.step-divider::before {
    content: '';
    display: block;
    border-top: 1px solid #d1d9e0;
    margin-bottom: -8px;
}
.message {
    margin-bottom: 12px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #d1d9e0;
}
.message-header {
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.message-body {
    padding: 12px 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
/* System */
.msg-system .message-header { background: #eef1f4; color: #59636e; }
.msg-system .message-body { background: #f6f8fa; color: #59636e; font-size: 12px; }
/* User */
.msg-user .message-header { background: #ddf4ff; color: #0550ae; }
.msg-user .message-body { background: #f0f7ff; color: #1f2328; }
/* Assistant */
.msg-assistant .message-header { background: #dafbe1; color: #1a7f37; }
.msg-assistant .message-body { background: #f0fff2; color: #1f2328; }
/* Reasoning */
.reasoning-block {
    margin: 0 14px 8px 14px;
    padding: 10px 12px;
    background: #f5f0ff;
    border-left: 3px solid #8250df;
    border-radius: 4px;
    color: #3d2b5a;
    font-size: 12px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.reasoning-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8250df;
    margin: 10px 14px 4px 14px;
}
/* Tool calls */
.tool-call {
    margin: 8px 14px;
    padding: 8px 12px;
    background: #fff8f0;
    border-left: 3px solid #bc4c00;
    border-radius: 4px;
    font-size: 12px;
}
.tool-call-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #bc4c00;
    margin: 10px 14px 4px 14px;
}
.tool-call-name { color: #bc4c00; font-weight: 600; }
.tool-call-args {
    color: #3d3d3d;
    margin-top: 4px;
    white-space: pre-wrap;
    word-wrap: break-word;
}
/* Tool response */
.msg-tool .message-header { background: #fff1e5; color: #bc4c00; }
.msg-tool .message-body {
    background: #fffbf5;
    color: #3d2b5a;
}
"""


def _esc(text: str | None) -> str:
    if not text:
        return ""
    return html.escape(text)


def _format_command(arguments_json: str) -> str:
    """Extract and format the command from tool call arguments."""
    try:
        args = json.loads(arguments_json)
        if isinstance(args, dict) and "command" in args:
            return args["command"]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return json.dumps(json.loads(arguments_json), indent=2)
    except (json.JSONDecodeError, TypeError):
        return arguments_json


def render_message(msg: dict, step_num: int | None = None) -> str:
    """Render a single message as an HTML block."""
    role = msg.get("role", "unknown")
    css_class = f"msg-{role}"

    if role == "system":
        return f'''<div class="message {css_class}">
  <div class="message-header">System</div>
  <div class="message-body">{_esc(msg.get("content"))}</div>
</div>'''

    if role == "user":
        content = msg.get("content", "")
        return f'''<div class="message {css_class}">
  <div class="message-header">User</div>
  <div class="message-body">{_esc(content)}</div>
</div>'''

    if role == "tool":
        tool_id = msg.get("tool_call_id", "")
        content = msg.get("content", "")
        return f'''<div class="message {css_class}">
  <div class="message-header">Tool Result — {_esc(tool_id)}</div>
  <div class="message-body">{_esc(content)}</div>
</div>'''

    if role == "assistant":
        parts = []

        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if reasoning:
            char_count = len(reasoning)
            parts.append(f'''  <div class="reasoning-label">Reasoning ({char_count:,} chars)</div>
  <div class="reasoning-block">{_esc(reasoning)}</div>''')

        content = msg.get("content") or ""
        if content:
            parts.append(f'  <div class="message-body">{_esc(content)}</div>')

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "unknown")
            fn_args = fn.get("arguments", "")
            formatted = _format_command(fn_args)
            parts.append(f'''  <div class="tool-call-label">Tool Call — {_esc(fn_name)}</div>
  <div class="tool-call">
    <div class="tool-call-args">{_esc(formatted)}</div>
  </div>''')

        inner = "\n".join(parts)
        step_label = f" — Step {step_num}" if step_num is not None else ""
        return f'''<div class="message {css_class}">
  <div class="message-header">Assistant{step_label}</div>
{inner}
</div>'''

    return f'''<div class="message">
  <div class="message-header">{_esc(role)}</div>
  <div class="message-body">{_esc(msg.get("content", ""))}</div>
</div>'''


def render_html(messages: list[dict], title: str = "", state: dict | None = None) -> str:
    """Render a full list of messages as an HTML page."""
    blocks = []
    step = 0

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            blocks.append(
                f'<div class="step-divider"><span>Step {step}</span></div>'
            )
            blocks.append(render_message(msg, step_num=step))
            step += 1
        else:
            blocks.append(render_message(msg))

    meta_parts = []
    if state:
        tr = state.get("test_results", {})
        if tr:
            passed = tr.get("passed", 0)
            total = tr.get("total", 0)
            badge_cls = "badge-pass" if tr.get("all_passed") else "badge-fail"
            meta_parts.append(
                f'Tests: {passed}/{total} <span class="badge {badge_cls}">'
                f'{"PASS" if tr.get("all_passed") else "FAIL"}</span>'
            )
        meta_parts.append(f'Steps: {state.get("step", "?")}')
    meta_html = " &nbsp;|&nbsp; ".join(meta_parts)

    body = "\n".join(blocks)
    page_title = title or "Rollout"
    heading = title or "Rollout"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(page_title)}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{_esc(heading)}</h1>
<div class="meta">{meta_html}</div>
{body}
</body>
</html>"""


def write_rollout_html(
    path: Path | str,
    output: Path | str | None = None,
) -> tuple[Path, int]:
    """Write ``rollout.html`` for a single run (or path to messages.json).

    Returns:
        (output_path, number of messages).
    """
    p = Path(path)
    mj_path, messages = resolve_single_rollout_messages(p)
    state = load_state(mj_path)
    run_name = mj_path.parent.parent.name
    title = f"{run_name} — {state.get('task_name', 'rollout')}"
    html_out = render_html(messages, title=title, state=state)
    out_path = Path(output) if output else mj_path.parent.parent / "rollout.html"
    out_path.write_text(html_out)
    return out_path, len(messages)


def main(path: str, output: str | None = None) -> None:
    """Render rollout(s) as HTML.

    Args:
        path: A run-* directory, messages.json, or a timestamp dir with run-* children.
        output: Output path. Only allowed for a single rollout; default is
            rollout.html next to that run directory. For timestamp dirs, each run
            always gets ``run-*/rollout.html``.
    """
    p = Path(path)
    entries = list_rollout_entries(p)
    if len(entries) > 1:
        if output is not None:
            raise ValueError("'output' is only supported when path resolves to a single rollout")
        for mj_path, _ in entries:
            rd = mj_path.parent.parent
            out_path, n = write_rollout_html(rd)
            print(f"  Wrote {out_path} ({n} messages)")
        print(f"\nRendered {len(entries)} rollouts under {p}")
        return
    out_path, n = write_rollout_html(p, output)
    print(f"Wrote {out_path} ({n} messages)")


if __name__ == "__main__":
    fire.Fire(main)
