// SPDX-License-Identifier: Apache-2.0

package api

import (
	"bufio"
	"strconv"
	"sync"
)

// SSE framing constants.
var (
	ssePrefix = []byte("data: ")
	sseEnd    = []byte("\n\n")
	sseDone   = []byte("data: [DONE]\n\n")
)

// chunkBufPool recycles encode buffers so streaming does not allocate per chunk.
var chunkBufPool = sync.Pool{New: func() any { return make([]byte, 0, 512) }}

// StreamEncoder writes OpenAI chat-completion SSE chunks with no per-chunk heap
// allocation on the hot path. It hand-builds the JSON for the common
// content-delta chunk shape; the rare role/finish/usage chunks reuse the same
// pooled buffer. Ported from the zero-alloc design in api/streaming.py.
type StreamEncoder struct {
	w        *bufio.Writer
	id       string
	model    string
	created  int64
	roleSent bool
}

// NewStreamEncoder constructs an encoder bound to a buffered writer.
func NewStreamEncoder(w *bufio.Writer, id, model string, created int64) *StreamEncoder {
	return &StreamEncoder{w: w, id: id, model: model, created: created}
}

// header writes the shared chunk envelope opener into buf.
func (e *StreamEncoder) header(buf []byte) []byte {
	buf = append(buf, `{"id":`...)
	buf = appendJSONString(buf, e.id)
	buf = append(buf, `,"object":"chat.completion.chunk","created":`...)
	buf = strconv.AppendInt(buf, e.created, 10)
	buf = append(buf, `,"model":`...)
	buf = appendJSONString(buf, e.model)
	buf = append(buf, `,"choices":[{"index":0,"delta":`...)
	return buf
}

// Role emits the opening chunk carrying the assistant role. It is a no-op after
// the first call.
func (e *StreamEncoder) Role() error {
	if e.roleSent {
		return nil
	}
	e.roleSent = true
	buf := chunkBufPool.Get().([]byte)[:0]
	buf = e.header(buf)
	buf = append(buf, `{"role":"assistant"},"finish_reason":null}]}`...)
	err := e.frame(buf)
	chunkBufPool.Put(buf)
	return err
}

// Content emits a content-delta chunk. The reasoning flag routes the text to
// the reasoning_content field instead.
func (e *StreamEncoder) Content(text string, reasoning bool) error {
	if text == "" {
		return nil
	}
	buf := chunkBufPool.Get().([]byte)[:0]
	buf = e.header(buf)
	if reasoning {
		buf = append(buf, `{"reasoning_content":`...)
	} else {
		buf = append(buf, `{"content":`...)
	}
	buf = appendJSONString(buf, text)
	buf = append(buf, `},"finish_reason":null}]}`...)
	err := e.frame(buf)
	chunkBufPool.Put(buf)
	return err
}

// DeltaJSON emits a chunk whose delta is the caller-provided JSON object (for
// example a tool-call delta the tool parser produced). The bytes must be a
// complete JSON object such as {"tool_calls":[...]}; the envelope and trailing
// finish_reason are added here.
func (e *StreamEncoder) DeltaJSON(deltaJSON []byte) error {
	buf := chunkBufPool.Get().([]byte)[:0]
	buf = e.header(buf)
	buf = append(buf, deltaJSON...)
	buf = append(buf, `,"finish_reason":null}]}`...)
	err := e.frame(buf)
	chunkBufPool.Put(buf)
	return err
}

// Finish emits the terminal chunk with a finish_reason and optional usage.
func (e *StreamEncoder) Finish(reason string, usage *Usage) error {
	buf := chunkBufPool.Get().([]byte)[:0]
	buf = e.header(buf)
	buf = append(buf, `{},"finish_reason":`...)
	buf = appendJSONString(buf, reason)
	buf = append(buf, `}]`...)
	if usage != nil {
		buf = append(buf, `,"usage":{"prompt_tokens":`...)
		buf = strconv.AppendInt(buf, int64(usage.PromptTokens), 10)
		buf = append(buf, `,"completion_tokens":`...)
		buf = strconv.AppendInt(buf, int64(usage.CompletionTokens), 10)
		buf = append(buf, `,"total_tokens":`...)
		buf = strconv.AppendInt(buf, int64(usage.TotalTokens), 10)
		buf = append(buf, '}')
	}
	buf = append(buf, '}')
	err := e.frame(buf)
	chunkBufPool.Put(buf)
	return err
}

// Done writes the terminating [DONE] sentinel.
func (e *StreamEncoder) Done() error {
	_, err := e.w.Write(sseDone)
	if err != nil {
		return err
	}
	return e.w.Flush()
}

// frame writes a single SSE event (data: <json>\n\n) and flushes.
func (e *StreamEncoder) frame(payload []byte) error {
	if _, err := e.w.Write(ssePrefix); err != nil {
		return err
	}
	if _, err := e.w.Write(payload); err != nil {
		return err
	}
	if _, err := e.w.Write(sseEnd); err != nil {
		return err
	}
	return e.w.Flush()
}

// appendJSONString appends a JSON-quoted, escaped string to dst without
// allocating an intermediate string. Escapes the characters JSON requires plus
// control bytes, matching encoding/json's HTML-safe defaults turned off (we do
// not escape <, >, &, since OpenAI SSE consumers expect raw).
func appendJSONString(dst []byte, s string) []byte {
	dst = append(dst, '"')
	start := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 0x20 && c != '"' && c != '\\' {
			continue
		}
		dst = append(dst, s[start:i]...)
		switch c {
		case '"':
			dst = append(dst, '\\', '"')
		case '\\':
			dst = append(dst, '\\', '\\')
		case '\n':
			dst = append(dst, '\\', 'n')
		case '\r':
			dst = append(dst, '\\', 'r')
		case '\t':
			dst = append(dst, '\\', 't')
		default:
			const hex = "0123456789abcdef"
			dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xf])
		}
		start = i + 1
	}
	dst = append(dst, s[start:]...)
	dst = append(dst, '"')
	return dst
}
