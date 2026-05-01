"""
HTML visualization utilities for priming experiments.

This module generates interactive HTML visualizations showing how
different primes affect token-level log probabilities.
"""

import html
from pathlib import Path


def diff_to_color(diff: float, max_abs: float) -> str:
    """
    Convert logprob difference to RGB color string.

    Uses a lighter color palette so text remains readable.

    Args:
        diff: The logprob difference (primed - baseline).
        max_abs: Maximum absolute difference for normalization.

    Returns:
        RGB color string.
    """
    if max_abs == 0:
        return "rgb(255, 255, 255)"

    # Normalize to [-1, 1]
    normalized = max(-1, min(1, diff / max_abs))

    # Use a minimum intensity of 150 so colors aren't too dark
    min_intensity = 150

    if normalized > 0:
        # Red/pink for positive (more likely with prime)
        intensity = int(255 - (255 - min_intensity) * normalized)
        return f"rgb(255, {intensity}, {intensity})"
    else:
        # Blue/light blue for negative (less likely with prime)
        intensity = int(255 - (255 - min_intensity) * (-normalized))
        return f"rgb({intensity}, {intensity}, 255)"


def tokens_to_sentences(tokens: list[str], logprobs: list[float]) -> list[dict]:
    """
    Group tokens into sentences (split on newlines).

    Args:
        tokens: List of token strings.
        logprobs: List of log probabilities for each token.

    Returns:
        List of dicts with:
            - text: the sentence text
            - tokens: list of tokens in the sentence
            - logprobs: list of logprobs for each token
            - total_logprob: sum of logprobs (log P(sentence))
    """
    sentences = []
    current_tokens = []
    current_logprobs = []
    current_text = ""

    for token, lp in zip(tokens, logprobs):
        current_tokens.append(token)
        current_logprobs.append(lp)
        current_text += token

        # Split on newline
        if "\n" in token:
            sentences.append(
                {
                    "text": current_text,
                    "tokens": current_tokens,
                    "logprobs": current_logprobs,
                    "total_logprob": sum(p for p in current_logprobs if p is not None),
                }
            )
            current_tokens = []
            current_logprobs = []
            current_text = ""

    # Don't forget the last sentence
    if current_tokens:
        sentences.append(
            {
                "text": current_text,
                "tokens": current_tokens,
                "logprobs": current_logprobs,
                "total_logprob": sum(p for p in current_logprobs if p is not None),
            }
        )

    return sentences


def generate_html_visualization(
    tokens: list[str],
    diff_logprobs: list[float],
    baseline_logprobs: list[float],
    primed_logprobs: list[float],
    prime_name: str,
    output_path: Path,
    mode: str = "token",
):
    """
    Generate an HTML visualization of logprob differences.

    Args:
        tokens: List of token strings.
        diff_logprobs: Difference in logprobs (primed - baseline).
        baseline_logprobs: Baseline logprobs.
        primed_logprobs: Primed logprobs.
        prime_name: Name of the prime for the title.
        output_path: Where to save the HTML file.
        mode: "token" for per-token coloring, "sentence" for per-line coloring
              with sum of logprobs (log P(sentence)).

    Colors:
        - Red = more likely with prime (positive diff)
        - Blue = less likely with prime (negative diff)
        - Hover shows exact values
    """
    if mode == "sentence":
        # Group into sentences and compute sentence-level diffs
        baseline_sentences = tokens_to_sentences(tokens, baseline_logprobs)
        primed_sentences = tokens_to_sentences(tokens, primed_logprobs)

        sentence_diffs = []
        for bs, ps in zip(baseline_sentences, primed_sentences):
            diff = ps["total_logprob"] - bs["total_logprob"]
            sentence_diffs.append(
                {
                    "text": bs["text"],
                    "baseline_logprob": bs["total_logprob"],
                    "primed_logprob": ps["total_logprob"],
                    "diff": diff,
                    "n_tokens": len(bs["tokens"]),
                }
            )

        # Calculate max absolute value for color scaling
        valid_diffs = [s["diff"] for s in sentence_diffs if s["diff"] != 0]
        if valid_diffs:
            max_abs = max(abs(min(valid_diffs)), abs(max(valid_diffs)))
        else:
            max_abs = 1.0

        # Build sentence rows with inline metadata
        content_spans = []
        for i, s in enumerate(sentence_diffs):
            color = diff_to_color(s["diff"], max_abs)

            # Display text (strip trailing newline for cleaner display)
            display_text = s["text"].rstrip("\n")
            display_text = html.escape(display_text)

            # Format: [line_num] (diff) sentence_text
            diff_sign = "+" if s["diff"] >= 0 else ""
            row = (
                f'<div class="sentence-row">'
                f'<span class="meta">[{i:3d}] ({diff_sign}{s["diff"]:.2f})</span> '
                f'<span class="sentence" style="background-color: {color};">'
                f"{display_text}</span></div>"
            )
            content_spans.append(row)

        stats_html = (
            f"<strong>Stats:</strong> {len(sentence_diffs)} lines | "
            f"Max diff: {max(s['diff'] for s in sentence_diffs):.4f} | "
            f"Min diff: {min(s['diff'] for s in sentence_diffs):.4f} | "
            f"Mean diff: {sum(s['diff'] for s in sentence_diffs)/len(sentence_diffs):.4f}"
        )

        mode_label = (
            "Sentence-level (sum of log probs per line). Format: [line] (delta) text"
        )

    else:  # token mode
        # Calculate max absolute value for color scaling
        valid_diffs = [d for d in diff_logprobs if d is not None and d != 0]
        if valid_diffs:
            max_abs = max(abs(min(valid_diffs)), abs(max(valid_diffs)))
        else:
            max_abs = 1.0

        # Build token spans
        content_spans = []
        for i, (token, diff, bl, pl) in enumerate(
            zip(tokens, diff_logprobs, baseline_logprobs, primed_logprobs)
        ):
            color = diff_to_color(diff, max_abs)
            escaped_token = html.escape(repr(token)[1:-1])

            bl_str = f"{bl:.4f}" if bl is not None else "N/A"
            pl_str = f"{pl:.4f}" if pl is not None else "N/A"
            diff_str = f"{diff:+.4f}" if diff is not None else "N/A"

            tooltip = (
                f"Token {i}: {escaped_token}&#10;"
                f"Baseline: {bl_str}&#10;"
                f"Primed: {pl_str}&#10;"
                f"Diff: {diff_str}"
            )

            display_token = (
                token.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "\u00b7")
            )
            display_token = html.escape(display_token)

            span = (
                f'<span class="token" style="background-color: {color};" '
                f'title="{tooltip}">{display_token}</span>'
            )
            content_spans.append(span)

        stats_html = (
            f"<strong>Stats:</strong> {len(tokens)} tokens | "
            f"Max diff: {max(diff_logprobs):.4f} | "
            f"Min diff: {min(diff_logprobs):.4f} | "
            f"Mean diff: {sum(diff_logprobs)/len(diff_logprobs):.4f}"
        )

        mode_label = "Token-level"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Logprob Analysis: {html.escape(prime_name)}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
            padding: 20px;
            max-width: 100%;
            background: #1a1a1a;
            color: #fff;
        }}
        h1 {{
            font-family: Arial, sans-serif;
            color: #fff;
        }}
        .mode-label {{
            color: #888;
            margin-bottom: 10px;
        }}
        .legend {{
            margin: 20px 0;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        .content {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 5px;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.8;
        }}
        .token, .sentence {{
            padding: 2px 4px;
            border-radius: 2px;
            cursor: pointer;
            color: #000;
        }}
        .token:hover, .sentence:hover {{
            outline: 2px solid yellow;
        }}
        .sentence-row {{
            margin: 2px 0;
            display: flex;
            align-items: flex-start;
        }}
        .meta {{
            color: #888;
            font-size: 12px;
            min-width: 140px;
            flex-shrink: 0;
        }}
        .stats {{
            margin: 20px 0;
            padding: 10px;
            background: #2a2a2a;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Logprob Difference: {html.escape(prime_name)}</h1>
    <div class="mode-label">{mode_label}</div>

    <div class="legend">
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(255, 150, 150);"></div>
            <span>More likely with prime (+)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(255, 255, 255);"></div>
            <span>No change (0)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(150, 150, 255);"></div>
            <span>Less likely with prime (-)</span>
        </div>
    </div>

    <div class="stats">{stats_html}</div>

    <div class="content">{''.join(content_spans)}</div>

    <script>
        document.querySelectorAll('.token, .sentence').forEach(el => {{
            el.addEventListener('click', () => console.log(el.title.replace(/&#10;/g, '\\n')));
        }});
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved HTML to {output_path}")
