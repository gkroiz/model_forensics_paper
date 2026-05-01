"""
Utility functions for repeated resampling.
"""

import re


def extract_last_assistant_reasoning(messages: list[dict]) -> tuple[dict | None, int | None, str]:
    """
    Extract the last assistant turn and its reasoning from a conversation.
    
    Returns:
        (last_assistant_msg, index, reasoning_text)
    """
    last_turn, last_idx, reasoning = None, None, ""
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_turn, last_idx = msg, i
            reasoning = msg.get("reasoning") or msg.get("reasoning_content", "")
    return last_turn, last_idx, reasoning


def split_text_to_sentences(text: str, min_length: int = 4) -> tuple[list[str], list[int]]:
    """
    Split text into sentences with position tracking.
    
    Args:
        text: Text to split
        min_length: Minimum sentence length (shorter ones get merged)
    
    Returns:
        (sentences, positions) where positions[i] is the char offset of sentences[i]
    """
    if not text:
        return [], []

    # Protect abbreviations from being split
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'i.e.', 'e.g.', 'etc.', 'vs.', 'U.S.', 'U.K.']
    protected = text
    abbrev_map = {}
    for i, abbrev in enumerate(abbrevs):
        if abbrev in protected:
            placeholder = f"_AB{i:02d}_"
            abbrev_map[placeholder] = abbrev
            protected = protected.replace(abbrev, placeholder)

    # Find sentence boundaries
    boundaries = [0]
    for m in re.finditer(r'[.!?]\s+([A-Z])', protected):
        boundaries.append(m.start(1))
    for m in re.finditer(r'\n', protected):
        prev_pos = m.start() - 1
        if prev_pos >= 0 and protected[prev_pos] not in '.!?:':
            pos = m.end()
            while pos < len(protected) and protected[pos] in ' \t':
                pos += 1
            if pos < len(protected) and protected[pos] != '\n':
                boundaries.append(pos)
    boundaries.append(len(protected))
    boundaries = sorted(set(boundaries))

    # Extract segments
    segments = []
    for i in range(len(boundaries) - 1):
        segment_text = protected[boundaries[i]:boundaries[i+1]]
        for placeholder, abbrev in abbrev_map.items():
            segment_text = segment_text.replace(placeholder, abbrev)
        segments.append({'text': segment_text, 'position': boundaries[i], 'length': len(segment_text.strip())})

    # Merge short segments
    merged = []
    for seg in segments:
        if seg['length'] < min_length:
            if merged:
                merged[-1]['text'] += seg['text']
            else:
                merged.append(seg)
        else:
            merged.append(seg)

    sentences = [s['text'] for s in merged]
    positions = []
    pos = 0
    for s in merged:
        positions.append(pos)
        pos += len(s['text'])
    return sentences, positions
