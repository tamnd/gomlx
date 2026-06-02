// SPDX-License-Identifier: Apache-2.0

// Package compute holds the inference backend: model loading, samplers, logits
// processors, KV caches, and the batched generator. The numeric core here is
// pure Go and runs without a GPU; the on-device forward pass lives behind the
// "mlx" build tag and links against mlx-c. Files without a build tag compile
// everywhere, including the Linux CI that has no MLX installed.
package compute

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"syscall"
)

// Dtype is a safetensors element type. The string values match the safetensors
// spec exactly so they round-trip through the header JSON.
type Dtype string

const (
	F64  Dtype = "F64"
	F32  Dtype = "F32"
	F16  Dtype = "F16"
	BF16 Dtype = "BF16"
	I64  Dtype = "I64"
	I32  Dtype = "I32"
	I16  Dtype = "I16"
	I8   Dtype = "I8"
	U64  Dtype = "U64"
	U32  Dtype = "U32"
	U16  Dtype = "U16"
	U8   Dtype = "U8"
	BOOL Dtype = "BOOL"
)

// Size returns the byte width of one element, or 0 for an unknown dtype.
func (d Dtype) Size() int {
	switch d {
	case F64, I64, U64:
		return 8
	case F32, I32, U32:
		return 4
	case F16, BF16, I16, U16:
		return 2
	case I8, U8, BOOL:
		return 1
	default:
		return 0
	}
}

// TensorInfo describes one tensor in a safetensors file. Begin and End are byte
// offsets into the data section (the bytes after the JSON header), so they are
// independent of where the file is mapped.
type TensorInfo struct {
	Name  string
	Dtype Dtype
	Shape []int
	Begin int64
	End   int64
}

// NumElements returns the product of the shape dimensions.
func (t TensorInfo) NumElements() int64 {
	n := int64(1)
	for _, d := range t.Shape {
		n *= int64(d)
	}
	return n
}

// nbytes returns the declared byte length of the tensor payload.
func (t TensorInfo) nbytes() int64 { return t.End - t.Begin }

// Header is the parsed safetensors header: the per-tensor metadata, the tensor
// names in file order, and the optional free-form __metadata__ map.
type Header struct {
	Tensors  map[string]TensorInfo
	Order    []string
	Metadata map[string]string
}

// headerEntry is the JSON shape of one tensor record in the header.
type headerEntry struct {
	Dtype       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

// ParseHeader decodes a safetensors header (the JSON object that follows the
// 8-byte length prefix). It preserves the order tensors appear in the file and
// validates that each record carries a two-element data_offsets pair.
func ParseHeader(header []byte) (*Header, error) {
	dec := json.NewDecoder(bytes.NewReader(header))
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("safetensors: header is not a JSON object")
	}

	h := &Header{Tensors: make(map[string]TensorInfo)}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("safetensors: read key: %w", err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, fmt.Errorf("safetensors: non-string header key")
		}
		if key == "__metadata__" {
			var md map[string]string
			if err := dec.Decode(&md); err != nil {
				return nil, fmt.Errorf("safetensors: decode __metadata__: %w", err)
			}
			h.Metadata = md
			continue
		}
		var e headerEntry
		if err := dec.Decode(&e); err != nil {
			return nil, fmt.Errorf("safetensors: decode %q: %w", key, err)
		}
		if len(e.DataOffsets) != 2 {
			return nil, fmt.Errorf("safetensors: %q has %d data_offsets, want 2", key, len(e.DataOffsets))
		}
		if e.DataOffsets[1] < e.DataOffsets[0] {
			return nil, fmt.Errorf("safetensors: %q has reversed data_offsets", key)
		}
		h.Order = append(h.Order, key)
		h.Tensors[key] = TensorInfo{
			Name:  key,
			Dtype: Dtype(e.Dtype),
			Shape: e.Shape,
			Begin: e.DataOffsets[0],
			End:   e.DataOffsets[1],
		}
	}
	return h, nil
}

// readHeaderLen reads the 8-byte little-endian header length prefix.
func readHeaderLen(prefix []byte) (uint64, error) {
	if len(prefix) < 8 {
		return 0, fmt.Errorf("safetensors: file shorter than 8-byte length prefix")
	}
	return binary.LittleEndian.Uint64(prefix[:8]), nil
}

// SafeTensors is an opened safetensors file. The data section is memory-mapped,
// so per-tensor byte slices are views into the mapping with no copy. Call Close
// to unmap.
type SafeTensors struct {
	Header *Header

	mmap      []byte // whole-file image (mmap from Open, or caller bytes)
	dataStart int64  // offset of the data section within the file
	mapped    bool   // true when mmap is a real syscall mapping we must unmap
}

// Open maps a safetensors file and parses its header. The returned value holds
// a mapping of the whole file until Close is called.
func Open(path string) (*SafeTensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := fi.Size()
	if size < 8 {
		return nil, fmt.Errorf("safetensors: %s is too small", path)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("safetensors: mmap %s: %w", path, err)
	}

	st, err := fromBytes(data)
	if err != nil {
		_ = syscall.Munmap(data)
		return nil, err
	}
	st.mapped = true
	return st, nil
}

// FromBytes parses an in-memory safetensors image. It is the copy-free path
// used by tests and by callers that already hold the bytes; Close is a no-op.
func FromBytes(data []byte) (*SafeTensors, error) {
	return fromBytes(data)
}

func fromBytes(data []byte) (*SafeTensors, error) {
	hlen, err := readHeaderLen(data)
	if err != nil {
		return nil, err
	}
	dataStart := int64(8) + int64(hlen)
	if dataStart > int64(len(data)) {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size", hlen)
	}
	h, err := ParseHeader(data[8:dataStart])
	if err != nil {
		return nil, err
	}

	st := &SafeTensors{Header: h, mmap: data, dataStart: dataStart}

	// Validate every tensor fits within the data section and that the declared
	// byte length matches shape*dtype. A mismatch means a corrupt or truncated
	// file, which we would rather catch here than as a wild slice later.
	dataLen := int64(len(data)) - dataStart
	for _, name := range h.Order {
		ti := h.Tensors[name]
		if ti.End > dataLen {
			return nil, fmt.Errorf("safetensors: %q extends past end of data section", name)
		}
		if sz := ti.Dtype.Size(); sz > 0 {
			if want := ti.NumElements() * int64(sz); want != ti.nbytes() {
				return nil, fmt.Errorf("safetensors: %q byte length %d != shape*dtype %d", name, ti.nbytes(), want)
			}
		}
	}
	return st, nil
}

// Bytes returns the raw little-endian bytes for a tensor as a view into the
// mapping. The slice is valid until Close.
func (s *SafeTensors) Bytes(name string) ([]byte, error) {
	ti, ok := s.Header.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	begin := s.dataStart + ti.Begin
	end := s.dataStart + ti.End
	return s.mmap[begin:end], nil
}

// Names returns the tensor names in file order.
func (s *SafeTensors) Names() []string { return s.Header.Order }

// Close unmaps the file. It is a no-op for a value created by FromBytes, whose
// backing slice the caller owns.
func (s *SafeTensors) Close() error {
	if !s.mapped || s.mmap == nil {
		s.mmap = nil
		return nil
	}
	err := syscall.Munmap(s.mmap)
	s.mmap = nil
	return err
}
