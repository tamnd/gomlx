// SPDX-License-Identifier: Apache-2.0

package toolparsers

import (
	"encoding/json"
	"strconv"
	"strings"
)

// The tool parsers reproduce Python's json.dumps output byte-for-byte for the
// `arguments` field of each tool call, because callers compare those strings
// verbatim. Two behaviours of the stdlib differ from Python and must be
// emulated here:
//
//   - separators: Python's default json.dumps uses ", " and ": " (with a
//     space); Go's encoding/json uses "," and ":".
//   - key order: Python dicts preserve insertion order; Go maps do not.
//
// We therefore decode JSON into an order-preserving value tree (omap for
// objects, []any for arrays, json.Number for numbers) and re-encode it with a
// Python-faithful serializer. ensure_ascii is configurable per parser.

// kv is one ordered object entry.
type kv struct {
	k string
	v any
}

// omap is an insertion-ordered JSON object.
type omap struct {
	items []kv
}

func newOmap() *omap { return &omap{} }

// set appends or overwrites a key, preserving first-insertion order (matching
// Python dict assignment semantics).
func (o *omap) set(k string, v any) {
	for i := range o.items {
		if o.items[i].k == k {
			o.items[i].v = v
			return
		}
	}
	o.items = append(o.items, kv{k, v})
}

func (o *omap) len() int { return len(o.items) }

// get returns the value for k, or nil when absent. A nil receiver returns nil
// so callers can probe an optional nested object without a guard.
func (o *omap) get(k string) any {
	if o == nil {
		return nil
	}
	for i := range o.items {
		if o.items[i].k == k {
			return o.items[i].v
		}
	}
	return nil
}

// has reports whether k is present, distinguishing a stored nil from absence.
func (o *omap) has(k string) bool {
	if o == nil {
		return false
	}
	for i := range o.items {
		if o.items[i].k == k {
			return true
		}
	}
	return false
}

// decodeOrdered parses JSON text into an order-preserving tree. Objects become
// *omap, arrays []any, numbers json.Number, mirroring Python's json.loads.
func decodeOrdered(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	// Reject trailing junk to match json.loads (which consumes the whole string).
	if dec.More() {
		return nil, errTrailing
	}
	return v, nil
}

var errTrailing = &json.SyntaxError{}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := newOmap()
			for dec.More() {
				keyTok, err := dec.Token()
				if err != nil {
					return nil, err
				}
				key := keyTok.(string)
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				o.set(key, val)
			}
			if _, err := dec.Token(); err != nil { // closing }
				return nil, err
			}
			return o, nil
		case '[':
			arr := []any{}
			for dec.More() {
				val, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, val)
			}
			if _, err := dec.Token(); err != nil { // closing ]
				return nil, err
			}
			return arr, nil
		}
	}
	return tok, nil // string, json.Number, bool, nil
}

// dumps serializes a value tree the way Python's json.dumps does by default
// (separators ", " and ": "). asciiOnly mirrors ensure_ascii=True.
func dumps(v any, asciiOnly bool) string {
	var b strings.Builder
	writeValue(&b, v, asciiOnly)
	return b.String()
}

func writeValue(b *strings.Builder, v any, ascii bool) {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		b.WriteString(pyQuote(x, ascii))
	case json.Number:
		b.WriteString(formatNumber(string(x)))
	case float64:
		b.WriteString(formatFloat(x))
	case int:
		b.WriteString(strconv.Itoa(x))
	case int64:
		b.WriteString(strconv.FormatInt(x, 10))
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			writeValue(b, e, ascii)
		}
		b.WriteByte(']')
	case *omap:
		b.WriteByte('{')
		for i, e := range x.items {
			if i > 0 {
				b.WriteString(", ")
			}
			b.WriteString(pyQuote(e.k, ascii))
			b.WriteString(": ")
			writeValue(b, e.v, ascii)
		}
		b.WriteByte('}')
	default:
		b.WriteString("null")
	}
}

// formatNumber renders a JSON numeric literal as Python's json.dumps would:
// integers pass through verbatim; anything with a fraction/exponent is
// normalized through float formatting.
func formatNumber(lit string) string {
	if !strings.ContainsAny(lit, ".eE") {
		return lit
	}
	f, err := strconv.ParseFloat(lit, 64)
	if err != nil {
		return lit
	}
	return formatFloat(f)
}

// formatFloat mimics Python's repr(float): shortest round-trip, but always with
// a fractional or exponent marker (1.0, not 1).
func formatFloat(f float64) string {
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// pyQuote quotes a string exactly as Python's json encoder does. With
// asciiOnly, code points above U+007F become \uXXXX (surrogate pairs above the
// BMP); otherwise they pass through literally. < > & / are never escaped.
func pyQuote(s string, asciiOnly bool) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r < 0x20:
				writeHex(&b, uint32(r))
			case asciiOnly && r > 0x7f:
				if r > 0xffff {
					r2 := uint32(r) - 0x10000
					writeHex(&b, 0xd800+(r2>>10))
					writeHex(&b, 0xdc00+(r2&0x3ff))
				} else {
					writeHex(&b, uint32(r))
				}
			default:
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}

const hexDigits = "0123456789abcdef"

func writeHex(b *strings.Builder, r uint32) {
	b.WriteString(`\u`)
	b.WriteByte(hexDigits[(r>>12)&0xf])
	b.WriteByte(hexDigits[(r>>8)&0xf])
	b.WriteByte(hexDigits[(r>>4)&0xf])
	b.WriteByte(hexDigits[r&0xf])
}

// jsonDumpValue is a convenience for parsers that already hold a decoded value
// (e.g. an arguments dict built from regex matches) and want Python-style
// output. asciiOnly defaults to false (ensure_ascii=False).
func jsonDumpValue(v any) string { return dumps(v, false) }
