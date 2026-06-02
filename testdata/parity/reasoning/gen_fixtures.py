# SPDX-License-Identifier: Apache-2.0
"""
Generate golden parity fixtures for the reasoning parsers.

This reads a reference Python implementation and records, for each case, the
result of the full extraction plus the chunk-by-chunk streaming path. The Go
parsers replay these fixtures and must produce identical splits.

Point the GOMLX_REASONING_REF environment variable at the reference module
(importable on PYTHONPATH) and run this script:

    GOMLX_REASONING_REF=<module> PYTHONPATH=<repo> \
        python3 testdata/parity/reasoning/gen_fixtures.py

Each fixture exercises both the full output and the delta-by-delta stream
(then finalize), capturing every emitted message so the comparison is exact.
"""

import importlib
import json
import os

REF = os.environ.get("GOMLX_REASONING_REF")
if not REF:
    raise SystemExit("set GOMLX_REASONING_REF to the reference reasoning module")
R = importlib.import_module(REF)

OUT = os.path.join(os.path.dirname(__file__), "fixtures.json")


def chunk(text, n):
    """Split text into n-ish character deltas (word-ish granularity)."""
    return [text[i : i + n] for i in range(0, len(text), n)]


# Each case: (parser_name, label, full_text, [stream_deltas]).
# stream_deltas defaults to a char-windowed split of full_text when None.
CASES = [
    # --- qwen3 ---
    ("qwen3", "both_tags", "<think>Let me analyze this.</think>The answer is 42.", None),
    ("qwen3", "implicit_close", "reasoning here</think>final answer", None),
    ("qwen3", "only_open_truncated", "<think>still thinking with no close", None),
    ("qwen3", "no_tags", "Just a plain answer.", None),
    ("qwen3", "tags_split", "<think>step one ", ["<think>", "step one ", "</think>", "done"]),
    ("qwen3", "injected_no_close", "thinking only no close tag", ["<think>", "thinking ", "only no close tag"]),
    # --- deepseek_r1 ---
    ("deepseek_r1", "both_tags", "<think>Step 1\nStep 2</think>The answer is 42.", None),
    ("deepseek_r1", "implicit_no_open", "reasoning content</think>final answer", None),
    ("deepseek_r1", "no_tags_short", "Hello there.", None),
    ("deepseek_r1", "no_tags_long", "This is a long piece of plain content that goes well beyond the sixty four character threshold for sure.", None),
    ("deepseek_r1", "close_in_delta_no_open", "abc</think>xyz", ["abc", "</think>", "xyz"]),
    ("deepseek_r1", "no_tags_short_stream", "hi there", ["hi ", "there"]),
    # --- glm4 ---
    ("glm4", "both_tags", "<think>reasoning</think>answer", None),
    ("glm4", "no_tags_content", "Plain GLM content with no thinking.", None),
    ("glm4", "box_markers", "<think>reason</think><|begin_of_box|>boxed answer<|end_of_box|>", None),
    ("glm4", "box_stream", "<|begin_of_box|>x<|end_of_box|>", ["<|begin_of_box|>", "x", "<|end_of_box|>"]),
    ("glm4", "no_tags_stream", "plain content streaming", ["plain ", "content ", "streaming"]),
    # --- gemma4 ---
    ("gemma4", "thought_and_content", "<|channel>thought\nthinking deeply<channel|><|channel>content\nthe reply<channel|>", None),
    ("gemma4", "no_channel", "just content no channels", None),
    ("gemma4", "thought_only", "<|channel>thought\nonly reasoning<channel|>", None),
    ("gemma4", "stream_flip", "<|channel>thought\nreason<channel|><|channel>content\nanswer<channel|>",
     ["<|channel>thought\n", "reason", "<channel|>", "<|channel>content\n", "answer", "<channel|>"]),
    # --- gpt_oss ---
    ("gpt_oss", "analysis_final", "<|channel|>analysis<|message|>thinking<|start|>assistant<|channel|>final<|message|>the answer<|return|>", None),
    ("gpt_oss", "no_channel", "plain content", None),
    ("gpt_oss", "constrain", "<|channel|>analysis<|message|>reason<|end|><|channel|>final <|constrain|>JSON<|message|>{\"a\":1}<|return|>", None),
    ("gpt_oss", "stream", "<|channel|>analysis<|message|>think<|channel|>final<|message|>answer<|return|>",
     ["<|channel|>analysis<|message|>", "think", "<|channel|>final<|message|>", "answer", "<|return|>"]),
    # --- harmony ---
    ("harmony", "analysis_final_return", "<|channel|>analysis<|message|>Thinking...<|end|><|channel|>final<|message|>Result.<|return|>", None),
    ("harmony", "final_end_terminator", "<|channel|>analysis<|message|>t<|end|><|channel|>final<|message|>answer<|end|>", None),
    ("harmony", "no_final", "<|channel|>analysis<|message|>just reasoning<|end|>", None),
    ("harmony", "stream", "<|channel|>analysis<|message|>think<|end|><|channel|>final<|message|>answer<|return|>",
     ["<|channel|>", "analysis", "<|message|>", "think", "<|end|>", "<|channel|>", "final", "<|message|>", "answer", "<|return|>"]),
    # --- minimax ---
    ("minimax", "explicit_think", "<think>let me reason</think>the answer", None),
    ("minimax", "reasoning_preamble", "The user asks for help.\n\nThe answer is 42 and more text here.", None),
    ("minimax", "direct_content", "```python\nprint(1)\n```", None),
    ("minimax", "plain_no_reasoning", "Hello, how can I help you today with your request?", None),
    ("minimax", "stream_reasoning", "I need to think about this carefully.\n\nThe answer is yes, definitely so here.",
     None),
    ("minimax", "stream_direct", "```code block here```", None),
]


def run_case(name, label, full, deltas):
    cls = R.get_parser(name)

    # Full extraction.
    p_full = cls()
    reasoning, content = p_full.extract_reasoning(full)

    # Streaming.
    p_stream = cls()
    p_stream.reset_state()
    if deltas is None:
        deltas = chunk(full, 6)
    emitted = []
    prev = ""
    for d in deltas:
        cur = prev + d
        msg = p_stream.extract_reasoning_streaming(prev, cur, d)
        if msg is not None:
            emitted.append({"reasoning": msg.reasoning, "content": msg.content})
        prev = cur
    fin = p_stream.finalize_streaming(prev)
    final_msg = None
    if fin is not None:
        final_msg = {"reasoning": fin.reasoning, "content": fin.content}

    return {
        "parser": name,
        "label": label,
        "full": full,
        "deltas": deltas,
        "extract": {"reasoning": reasoning, "content": content},
        "stream": emitted,
        "finalize": final_msg,
    }


def main():
    fixtures = [run_case(*c) for c in CASES]
    with open(OUT, "w") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(fixtures)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
