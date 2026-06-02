# SPDX-License-Identifier: Apache-2.0
"""
Generate golden parity fixtures for the tool-call parsers.

This reads a reference Python implementation and records, for each case, the
result of the full extraction plus the chunk-by-chunk streaming path. The Go
parsers replay these fixtures and must produce identical tool calls and stream
deltas.

Point GOMLX_TOOLPARSER_REF at the reference package that exposes a
ToolParserManager (importable on PYTHONPATH) and run this script:

    GOMLX_TOOLPARSER_REF=<module> PYTHONPATH=<repo> \
        python3 testdata/parity/toolparsers/gen_fixtures.py

Tool-call ids are random in both implementations, so they are stripped from
the recorded output; only names, arguments, content, and stream structure are
compared.

The reference package pulls in a couple of heavy optional dependencies only for
type hints and schema validation. When they are absent we inject light stubs so
the parsers themselves can run.
"""

import importlib
import json
import os
import sys
import types

# Light stubs for optional deps the parsers do not actually exercise here.
if "transformers" not in sys.modules:
    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        tr = types.ModuleType("transformers")
        tr.PreTrainedTokenizerBase = object
        sys.modules["transformers"] = tr
if "jsonschema" not in sys.modules:
    try:
        import jsonschema  # noqa: F401
    except ModuleNotFoundError:
        js = types.ModuleType("jsonschema")

        class _ValidationError(Exception):
            pass

        js.ValidationError = _ValidationError
        js.validate = lambda *a, **k: None
        sys.modules["jsonschema"] = js

REF = os.environ.get("GOMLX_TOOLPARSER_REF")
if not REF:
    raise SystemExit("set GOMLX_TOOLPARSER_REF to the reference tool-parser module")
R = importlib.import_module(REF)
MANAGER = R.ToolParserManager

OUT = os.path.join(os.path.dirname(__file__), "fixtures.json")


def chunk(text, n):
    """Split text into n-character deltas."""
    return [text[i : i + n] for i in range(0, len(text), n)]


# A tool schema used by the parsers that coerce parameter types.
WEATHER_TOOLS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                        "metric": {"type": "boolean"},
                    },
                },
            },
        }
    ]
}

# Each case: dict(parser, label, full, request=None, deltas=None).
# deltas defaults to a fixed-width split of full when omitted.
CASES = [
    # --- llama ---
    dict(parser="llama", label="single_call",
         full='<function=get_weather>{"city": "Paris"}</function>'),
    dict(parser="llama", label="text_then_call",
         full='Sure.<function=lookup>{"q": "go"}</function>'),
    dict(parser="llama", label="stream",
         full='<function=f>{"a": 1}</function>',
         deltas=['<function=f>', '{"a": 1}', '</function>']),
    dict(parser="llama", label="no_call", full="just a plain answer"),

    # --- granite ---
    dict(parser="granite", label="tool_call_marker",
         full='<|tool_call|>[{"name": "search", "arguments": {"q": "go"}}]'),
    dict(parser="granite", label="type_field",
         full='<tool_call>[{"type": "ping", "arguments": {}}]'),
    dict(parser="granite", label="stream",
         full='<|tool_call|>[{"name": "search", "arguments": {"q": "go"}}]',
         deltas=['<|tool_call|>[{"name": "search", ', '"arguments": {"q": "go"}}', ']']),

    # --- kimi ---
    dict(parser="kimi", label="single",
         full='<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'),
    dict(parser="kimi", label="text_before",
         full='Let me check.<|tool_call_begin|>functions.f:1<|tool_call_argument_begin|>{"a": 1}<|tool_call_end|>'),

    # --- nemotron ---
    dict(parser="nemotron", label="json_body",
         full='<tool_call><function=foo>{"a": 1}</function></tool_call>'),
    dict(parser="nemotron", label="param_body",
         full='<tool_call><function=bar><parameter=x>hello</parameter></function></tool_call>'),

    # --- deepseek (v3) ---
    dict(parser="deepseek", label="single",
         full='<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),

    # --- xlam ---
    dict(parser="xlam", label="code_fence",
         full='```json\n[{"name": "f", "arguments": {"a": 1}}]\n```'),
    dict(parser="xlam", label="tool_calls_tag",
         full='[TOOL_CALLS][{"name": "g", "arguments": {"b": 2}}]'),

    # --- qwen ---
    dict(parser="qwen", label="xml_json",
         full='<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'),
    dict(parser="qwen", label="bracket",
         full='[Calling tool: f({"a": 1})]'),
    dict(parser="qwen", label="stream_xml",
         full='<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>',
         deltas=['<tool_call>', '{"name": "f", ', '"arguments": {"a": 1}}', '</tool_call>']),

    # --- functionary ---
    dict(parser="functionary", label="recipient",
         full='<|recipient|>get_weather\n<|content|>{"city": "Paris"}'),

    # --- glm47 ---
    dict(parser="glm47", label="named",
         full='<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>',
         request=WEATHER_TOOLS),

    # --- gemma4 ---
    dict(parser="gemma4", label="call",
         full='<|tool_call>call:get_weather{city: "Paris"}<tool_call|>'),

    # --- mistral ---
    dict(parser="mistral", label="new_format",
         full='[TOOL_CALLS]get_weather{"city": "Paris"}'),
    dict(parser="mistral", label="old_array",
         full='[TOOL_CALLS][{"name": "f", "arguments": {"a": 1}}]'),

    # --- hermes ---
    dict(parser="hermes", label="xml_json",
         full='<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'),
    dict(parser="hermes", label="text_before",
         full='Sure thing.\n<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'),
    dict(parser="hermes", label="stream",
         full='<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>',
         deltas=['<tool_call>', '{"name": "f", "arguments": {"a": 1}}', '</tool_call>']),

    # --- minimax ---
    dict(parser="minimax", label="invoke",
         full='<minimax:tool_call><invoke name="get_weather"><parameter name="city">Paris</parameter></invoke></minimax:tool_call>'),

    # --- harmony ---
    dict(parser="harmony", label="commentary_final",
         full='<|channel|>commentary to=functions.get_weather<|message|>{"city": "Paris"}<|call|><|channel|>final<|message|>Here you go.<|return|>'),
    dict(parser="harmony", label="commentary_only",
         full='to=functions.f<|channel|>commentary<|message|>{"a": 1}<|call|>'),

    # --- deepseek_v31 ---
    dict(parser="deepseek_v31", label="single",
         full='<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Paris"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'),

    # --- seed_oss ---
    dict(parser="seed_oss", label="typed",
         full='<seed:tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function></seed:tool_call>',
         request=WEATHER_TOOLS),
    dict(parser="seed_oss", label="stream",
         full='<seed:tool_call><function=get_weather><parameter=city>Paris</parameter></function></seed:tool_call>',
         request=WEATHER_TOOLS,
         deltas=['<seed:tool_call>', '<function=get_weather>', '<parameter=city>Paris</parameter>', '</function>', '</seed:tool_call>']),

    # --- qwen3_coder_xml ---
    dict(parser="qwen3_coder_xml", label="typed",
         full='<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter><parameter=metric>true</parameter></function></tool_call>',
         request=WEATHER_TOOLS),
    dict(parser="qwen3_coder_xml", label="stream",
         full='<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>',
         request=WEATHER_TOOLS,
         deltas=['<tool_call>', '<function=get_weather>', '<parameter=city>Paris</parameter>', '</function>', '</tool_call>']),

    # --- auto ---
    dict(parser="auto", label="mistral",
         full='[TOOL_CALLS]f{"a": 1}'),
    dict(parser="auto", label="qwen_xml",
         full='<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>'),
    dict(parser="auto", label="llama",
         full='<function=f>{"a": 1}</function>'),
    dict(parser="auto", label="raw_json",
         full='{"name": "f", "arguments": {"a": 1}}'),
    dict(parser="auto", label="plain", full="no tool call here"),
]


def norm_calls(calls):
    """Keep only name and arguments; ids are random."""
    return [{"name": c["name"], "arguments": c["arguments"]} for c in calls]


def norm_stream(msg):
    """Strip ids/types; flatten function.name/arguments onto each entry."""
    out = {"content": msg.get("content"), "tool_calls": None}
    if "tool_calls" in msg and msg["tool_calls"] is not None:
        entries = []
        for tc in msg["tool_calls"]:
            fn = tc.get("function") or {}
            entries.append(
                {
                    "index": tc.get("index", 0),
                    "name": fn.get("name"),
                    "arguments": fn.get("arguments"),
                }
            )
        out["tool_calls"] = entries
    return out


def run_case(case):
    name = case["parser"]
    full = case["full"]
    request = case.get("request")
    deltas = case.get("deltas") or chunk(full, 6)

    cls = MANAGER.get_tool_parser(name)

    # Full extraction.
    parser = cls(tokenizer=None)
    info = parser.extract_tool_calls(full, request)
    extract = {
        "tools_called": info.tools_called,
        "tool_calls": norm_calls(info.tool_calls),
        "content": info.content,
    }

    # Streaming replay.
    sp = cls(tokenizer=None)
    sp.reset()
    stream = []
    prev = ""
    for d in deltas:
        cur = prev + d
        msg = sp.extract_tool_calls_streaming(prev, cur, d, request=request)
        if msg is not None:
            stream.append(norm_stream(msg))
        prev = cur

    return {
        "parser": name,
        "label": case["label"],
        "full": full,
        "request": request,
        "deltas": deltas,
        "extract": extract,
        "stream": stream,
    }


def main():
    fixtures = [run_case(c) for c in CASES]
    with open(OUT, "w") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
