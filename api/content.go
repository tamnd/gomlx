// SPDX-License-Identifier: Apache-2.0

package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// A chat message's content is either a plain string or an array of typed parts,
// the array form being how OpenAI carries multimodal input: text interleaved
// with images. The transport decodes the content into an untyped value, so these
// helpers normalize it into typed parts and pull the text and image references
// back out. Image references that arrive as data URLs are decoded to their
// bytes so a vision backend can consume them without re-parsing.

// ImageRef is one image referenced by a message. A data URL is decoded into
// Data with its MediaType filled in and IsData true; a plain http or https URL
// is left in URL with IsData false for the backend to fetch.
type ImageRef struct {
	URL       string
	MediaType string
	Data      []byte
	IsData    bool
}

// AudioRef is one audio clip carried by a message: the decoded bytes and the
// container format they arrived in, such as "wav" or "mp3". OpenAI sends audio
// inline as base64 rather than by URL, so Data is always present.
type AudioRef struct {
	Format string
	Data   []byte
}

// Parts normalizes the message content into typed parts. A nil content yields no
// parts; a string yields a single text part; an array is decoded part by part.
// Unknown part types are preserved with their Type set so a caller can decide
// what to do rather than having them silently dropped.
func (m Message) Parts() ([]ContentPart, error) {
	switch v := m.Content.(type) {
	case nil:
		return nil, nil
	case string:
		if v == "" {
			return nil, nil
		}
		return []ContentPart{{Type: "text", Text: v}}, nil
	default:
		// The content arrived as a decoded array of objects. Re-encoding and
		// decoding into the typed shape is robust to the concrete map types the
		// JSON decoder produced.
		raw, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("content: re-encode: %w", err)
		}
		var parts []ContentPart
		if err := json.Unmarshal(raw, &parts); err != nil {
			return nil, fmt.Errorf("content: not a string or array of parts: %w", err)
		}
		return parts, nil
	}
}

// TextContent returns the message text: a string content verbatim, or the text
// parts of an array content joined with single spaces. Non-text parts do not
// contribute. Malformed array content falls back to empty rather than erroring,
// matching how a server should treat content it cannot read as text.
func (m Message) TextContent() string {
	parts, err := m.Parts()
	if err != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" && p.Text != "" {
			if b.Len() > 0 {
				b.WriteByte(' ')
			}
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// ImageRefs returns the images referenced by the message, in order. Data URLs
// are decoded; other URLs are carried through for the backend to fetch. A part
// whose image_url is missing or empty is skipped. A data URL that does not
// decode is an error, since a truncated image is a client mistake worth
// reporting rather than dropping.
func (m Message) ImageRefs() ([]ImageRef, error) {
	parts, err := m.Parts()
	if err != nil {
		return nil, err
	}
	var refs []ImageRef
	for _, p := range parts {
		if p.Type != "image_url" || p.ImageURL == nil || p.ImageURL.URL == "" {
			continue
		}
		ref, err := parseImageURL(p.ImageURL.URL)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	return refs, nil
}

// HasImages reports whether the message carries any image parts, without
// decoding them. It lets a text-only path detect multimodal input cheaply.
func (m Message) HasImages() bool {
	parts, err := m.Parts()
	if err != nil {
		return false
	}
	for _, p := range parts {
		if p.Type == "image_url" && p.ImageURL != nil && p.ImageURL.URL != "" {
			return true
		}
	}
	return false
}

// AudioRefs returns the audio clips carried by the message, in order, decoding
// the base64 payload of each. A part whose input_audio is missing or has no data
// is skipped; a payload that does not decode is an error, since truncated audio
// is a client mistake worth reporting rather than dropping.
func (m Message) AudioRefs() ([]AudioRef, error) {
	parts, err := m.Parts()
	if err != nil {
		return nil, err
	}
	var refs []AudioRef
	for _, p := range parts {
		if p.Type != "input_audio" || p.InputAudio == nil || p.InputAudio.Data == "" {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(p.InputAudio.Data)
		if err != nil {
			return nil, fmt.Errorf("content: decode base64 audio: %w", err)
		}
		refs = append(refs, AudioRef{Format: p.InputAudio.Format, Data: data})
	}
	return refs, nil
}

// HasAudio reports whether the message carries any audio parts, without decoding
// them.
func (m Message) HasAudio() bool {
	parts, err := m.Parts()
	if err != nil {
		return false
	}
	for _, p := range parts {
		if p.Type == "input_audio" && p.InputAudio != nil && p.InputAudio.Data != "" {
			return true
		}
	}
	return false
}

// parseImageURL classifies an image URL. A data URL of the form
// data:[<mediatype>][;base64],<payload> is decoded into bytes; anything else is
// returned as a fetchable URL.
func parseImageURL(url string) (ImageRef, error) {
	if !strings.HasPrefix(url, "data:") {
		return ImageRef{URL: url}, nil
	}

	rest := url[len("data:"):]
	meta, payload, ok := strings.Cut(rest, ",")
	if !ok {
		return ImageRef{}, fmt.Errorf("content: data URL has no comma separator")
	}

	mediaType := meta
	isBase64 := false
	if before, after, ok := strings.Cut(meta, ";"); ok {
		mediaType = before
		isBase64 = strings.Contains(after, "base64")
	}

	var data []byte
	if isBase64 {
		decoded, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			return ImageRef{}, fmt.Errorf("content: decode base64 image: %w", err)
		}
		data = decoded
	} else {
		data = []byte(payload)
	}

	return ImageRef{
		URL:       url,
		MediaType: mediaType,
		Data:      data,
		IsData:    true,
	}, nil
}
