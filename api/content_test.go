// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"testing"
)

func TestPartsString(t *testing.T) {
	m := Message{Role: "user", Content: "hello world"}
	parts, err := m.Parts()
	if err != nil {
		t.Fatalf("parts: %v", err)
	}
	if len(parts) != 1 || parts[0].Type != "text" || parts[0].Text != "hello world" {
		t.Fatalf("string content should yield one text part, got %+v", parts)
	}
}

func TestPartsNilAndEmpty(t *testing.T) {
	if parts, err := (Message{}).Parts(); err != nil || parts != nil {
		t.Fatalf("nil content: got %v, %v", parts, err)
	}
	if parts, err := (Message{Content: ""}).Parts(); err != nil || parts != nil {
		t.Fatalf("empty string content: got %v, %v", parts, err)
	}
}

// decodeContent mimics the transport: a JSON body is decoded into an `any`, which
// is the exact shape Parts has to normalize.
func decodeContent(t *testing.T, body string) any {
	t.Helper()
	var m Message
	if err := json.Unmarshal([]byte(`{"role":"user","content":`+body+`}`), &m); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return m.Content
}

func TestPartsArrayFromJSON(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"text","text":"describe this"},
		{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
	]`)
	m := Message{Role: "user", Content: content}

	parts, err := m.Parts()
	if err != nil {
		t.Fatalf("parts: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("want 2 parts, got %d", len(parts))
	}
	if parts[0].Type != "text" || parts[0].Text != "describe this" {
		t.Errorf("part 0: %+v", parts[0])
	}
	if parts[1].Type != "image_url" || parts[1].ImageURL == nil || parts[1].ImageURL.URL != "https://example.com/a.png" {
		t.Errorf("part 1: %+v", parts[1])
	}
}

func TestTextContentJoinsTextParts(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"text","text":"first"},
		{"type":"image_url","image_url":{"url":"https://example.com/a.png"}},
		{"type":"text","text":"second"}
	]`)
	got := Message{Content: content}.TextContent()
	if got != "first second" {
		t.Errorf("TextContent: got %q want %q", got, "first second")
	}
}

func TestTextContentString(t *testing.T) {
	if got := (Message{Content: "verbatim"}).TextContent(); got != "verbatim" {
		t.Errorf("got %q want %q", got, "verbatim")
	}
}

func TestImageRefsHTTPURL(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"text","text":"x"},
		{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
	]`)
	refs, err := Message{Content: content}.ImageRefs()
	if err != nil {
		t.Fatalf("image refs: %v", err)
	}
	if len(refs) != 1 {
		t.Fatalf("want 1 ref, got %d", len(refs))
	}
	if refs[0].IsData || refs[0].URL != "https://example.com/a.png" || refs[0].Data != nil {
		t.Errorf("http ref should pass through undecoded: %+v", refs[0])
	}
}

func TestImageRefsDataURL(t *testing.T) {
	raw := []byte{0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02}
	enc := base64.StdEncoding.EncodeToString(raw)
	content := decodeContent(t, `[
		{"type":"image_url","image_url":{"url":"data:image/png;base64,`+enc+`"}}
	]`)
	refs, err := Message{Content: content}.ImageRefs()
	if err != nil {
		t.Fatalf("image refs: %v", err)
	}
	if len(refs) != 1 {
		t.Fatalf("want 1 ref, got %d", len(refs))
	}
	r := refs[0]
	if !r.IsData {
		t.Error("data URL should be flagged IsData")
	}
	if r.MediaType != "image/png" {
		t.Errorf("media type: got %q want image/png", r.MediaType)
	}
	if string(r.Data) != string(raw) {
		t.Errorf("decoded bytes: got %v want %v", r.Data, raw)
	}
}

func TestImageRefsBadDataURL(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"image_url","image_url":{"url":"data:image/png;base64,not valid base64!!"}}
	]`)
	if _, err := (Message{Content: content}).ImageRefs(); err == nil {
		t.Error("a malformed base64 data URL must error")
	}
}

func TestImageRefsDataURLNoComma(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"image_url","image_url":{"url":"data:image/png;base64"}}
	]`)
	if _, err := (Message{Content: content}).ImageRefs(); err == nil {
		t.Error("a data URL without a comma must error")
	}
}

func TestImageRefsSkipsEmpty(t *testing.T) {
	content := decodeContent(t, `[
		{"type":"image_url","image_url":{"url":""}},
		{"type":"image_url"},
		{"type":"text","text":"only text here"}
	]`)
	refs, err := Message{Content: content}.ImageRefs()
	if err != nil {
		t.Fatalf("image refs: %v", err)
	}
	if len(refs) != 0 {
		t.Errorf("empty and missing image urls should be skipped, got %d", len(refs))
	}
}

func TestHasImages(t *testing.T) {
	withImage := decodeContent(t, `[
		{"type":"text","text":"x"},
		{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
	]`)
	if !(Message{Content: withImage}).HasImages() {
		t.Error("content with an image should report HasImages true")
	}
	if (Message{Content: "plain text"}).HasImages() {
		t.Error("plain text should report HasImages false")
	}
}

func TestDataURLPlainText(t *testing.T) {
	// A data URL without base64 carries its payload verbatim.
	ref, err := parseImageURL("data:text/plain,hello")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !ref.IsData || ref.MediaType != "text/plain" || string(ref.Data) != "hello" {
		t.Errorf("plain data URL: %+v", ref)
	}
}
