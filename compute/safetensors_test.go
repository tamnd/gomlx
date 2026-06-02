// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// buildSafetensors assembles an in-memory safetensors image from ordered
// (name, dtype, shape, payload) records plus optional metadata.
func buildSafetensors(t *testing.T, metadata map[string]string, recs []struct {
	name    string
	dtype   string
	shape   []int
	payload []byte
}) []byte {
	t.Helper()
	var data bytes.Buffer
	var b bytes.Buffer
	b.WriteByte('{')
	first := true
	writeComma := func() {
		if !first {
			b.WriteByte(',')
		}
		first = false
	}
	if metadata != nil {
		writeComma()
		b.WriteString(`"__metadata__":{`)
		mfirst := true
		for k, v := range metadata {
			if !mfirst {
				b.WriteByte(',')
			}
			mfirst = false
			b.WriteString(`"` + k + `":"` + v + `"`)
		}
		b.WriteByte('}')
	}
	offset := 0
	for _, r := range recs {
		writeComma()
		b.WriteString(`"` + r.name + `":{"dtype":"` + r.dtype + `","shape":[`)
		for i, d := range r.shape {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(itoa(d))
		}
		end := offset + len(r.payload)
		b.WriteString(`],"data_offsets":[` + itoa(offset) + `,` + itoa(end) + `]}`)
		data.Write(r.payload)
		offset = end
	}
	b.WriteByte('}')

	header := b.Bytes()
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(header)))
	out = append(out, header...)
	out = append(out, data.Bytes()...)
	return out
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

func f32bytes(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func TestSafetensorsParseAndRead(t *testing.T) {
	w := f32bytes(1, 2, 3, 4, 5, 6) // 2x3
	bias := f32bytes(0.5, -0.5)     // 2
	img := buildSafetensors(t, map[string]string{"format": "pt"}, []struct {
		name    string
		dtype   string
		shape   []int
		payload []byte
	}{
		{"layer.weight", "F32", []int{2, 3}, w},
		{"layer.bias", "F32", []int{2}, bias},
	})

	st, err := FromBytes(img)
	if err != nil {
		t.Fatalf("FromBytes: %v", err)
	}
	defer st.Close()

	if got := st.Names(); len(got) != 2 || got[0] != "layer.weight" || got[1] != "layer.bias" {
		t.Fatalf("order: got %v", got)
	}
	if st.Header.Metadata["format"] != "pt" {
		t.Errorf("metadata: got %v", st.Header.Metadata)
	}

	wi := st.Header.Tensors["layer.weight"]
	if wi.Dtype != F32 || len(wi.Shape) != 2 || wi.Shape[0] != 2 || wi.Shape[1] != 3 {
		t.Errorf("weight info: %+v", wi)
	}
	if wi.NumElements() != 6 {
		t.Errorf("num elements: got %d", wi.NumElements())
	}

	gotW, err := st.Bytes("layer.weight")
	if err != nil {
		t.Fatalf("Bytes: %v", err)
	}
	if !bytes.Equal(gotW, w) {
		t.Errorf("weight bytes mismatch")
	}
	gotB, _ := st.Bytes("layer.bias")
	if !bytes.Equal(gotB, bias) {
		t.Errorf("bias bytes mismatch")
	}

	if _, err := st.Bytes("missing"); err == nil {
		t.Errorf("expected error for missing tensor")
	}
}

func TestSafetensorsDtypeSize(t *testing.T) {
	cases := map[Dtype]int{F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I8: 1, U8: 1, BOOL: 1, Dtype("WAT"): 0}
	for d, want := range cases {
		if got := d.Size(); got != want {
			t.Errorf("%s size: got %d want %d", d, got, want)
		}
	}
}

func TestSafetensorsRejectsBadLength(t *testing.T) {
	// Declared byte length disagrees with shape*dtype: 2x3 F32 should be 24 bytes.
	img := buildSafetensors(t, nil, []struct {
		name    string
		dtype   string
		shape   []int
		payload []byte
	}{
		{"w", "F32", []int{2, 3}, f32bytes(1, 2, 3)}, // only 12 bytes
	})
	if _, err := FromBytes(img); err == nil {
		t.Fatal("expected byte-length mismatch error")
	}
}

func TestSafetensorsRejectsTruncated(t *testing.T) {
	if _, err := FromBytes([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected error on short file")
	}
	// Header length prefix points past the end of the file.
	bad := make([]byte, 8)
	binary.LittleEndian.PutUint64(bad, 1000)
	if _, err := FromBytes(bad); err == nil {
		t.Fatal("expected error on oversized header length")
	}
}
