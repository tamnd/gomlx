// SPDX-License-Identifier: Apache-2.0

// Package audio turns the audio bytes carried by a multimodal request into the
// mono, fixed-rate float waveform a speech model consumes. A request delivers a
// clip as encoded bytes (WAV here); a speech encoder wants single-channel
// samples in [-1, 1] at a known sample rate, usually 16 kHz. This package does
// the decode, downmix, and resample on the host so the conversion is testable
// without a model.
package audio

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Waveform is a mono audio signal: samples in [-1, 1] tagged with the rate they
// were captured at, in hertz.
type Waveform struct {
	SampleRate int
	Samples    []float32
}

// WAV format tags from the fmt chunk: PCM is integer samples, IEEEFloat is
// 32-bit float samples. WAVE_FORMAT_EXTENSIBLE wraps one of these with a
// sub-format, which DecodeWAV resolves to the tag it extends.
const (
	formatPCM        = 0x0001
	formatIEEEFloat  = 0x0003
	formatExtensible = 0xFFFE
)

// DecodeWAV reads a RIFF/WAVE clip into a mono waveform, averaging channels and
// scaling integer samples into [-1, 1]. It handles the common encodings a client
// sends: 16- and 32-bit PCM and 32-bit IEEE float. An unsupported encoding or a
// malformed header is an error so a bad clip surfaces at decode rather than as
// silence downstream.
func DecodeWAV(data []byte) (*Waveform, error) {
	if len(data) < 12 || string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("audio: not a RIFF/WAVE file")
	}

	var (
		format        uint16
		channels      uint16
		sampleRate    uint32
		bitsPerSample uint16
		haveFmt       bool
		pcm           []byte
	)

	// Walk the chunk list after the 12-byte RIFF header, reading the fmt chunk for
	// the layout and the data chunk for the samples. Chunks are length-prefixed
	// and padded to an even size.
	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := pos + 8
		if body+size > len(data) {
			return nil, fmt.Errorf("audio: chunk %q runs past end of file", id)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return nil, fmt.Errorf("audio: fmt chunk too small")
			}
			format = binary.LittleEndian.Uint16(data[body : body+2])
			channels = binary.LittleEndian.Uint16(data[body+2 : body+4])
			sampleRate = binary.LittleEndian.Uint32(data[body+4 : body+8])
			bitsPerSample = binary.LittleEndian.Uint16(data[body+14 : body+16])
			if format == formatExtensible && size >= 26 {
				// The real format is the first two bytes of the sub-format GUID.
				format = binary.LittleEndian.Uint16(data[body+24 : body+26])
			}
			haveFmt = true
		case "data":
			pcm = data[body : body+size]
		}
		pos = body + size
		if size%2 == 1 {
			pos++ // chunks are word-aligned
		}
	}

	if !haveFmt {
		return nil, fmt.Errorf("audio: missing fmt chunk")
	}
	if pcm == nil {
		return nil, fmt.Errorf("audio: missing data chunk")
	}
	if channels == 0 {
		return nil, fmt.Errorf("audio: zero channels")
	}

	samples, err := decodeSamples(pcm, format, bitsPerSample)
	if err != nil {
		return nil, err
	}
	mono := downmix(samples, int(channels))
	return &Waveform{SampleRate: int(sampleRate), Samples: mono}, nil
}

// decodeSamples converts the raw data bytes into interleaved float samples in
// [-1, 1], dispatching on the format tag and bit depth.
func decodeSamples(pcm []byte, format, bits uint16) ([]float32, error) {
	switch {
	case format == formatPCM && bits == 16:
		n := len(pcm) / 2
		out := make([]float32, n)
		for i := range n {
			s := int16(binary.LittleEndian.Uint16(pcm[i*2 : i*2+2]))
			out[i] = float32(s) / 32768.0
		}
		return out, nil
	case format == formatPCM && bits == 32:
		n := len(pcm) / 4
		out := make([]float32, n)
		for i := range n {
			s := int32(binary.LittleEndian.Uint32(pcm[i*4 : i*4+4]))
			out[i] = float32(float64(s) / float64(math.MaxInt32+1))
		}
		return out, nil
	case format == formatIEEEFloat && bits == 32:
		n := len(pcm) / 4
		out := make([]float32, n)
		for i := range n {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(pcm[i*4 : i*4+4]))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("audio: unsupported format tag %d with %d-bit samples", format, bits)
	}
}

// downmix collapses interleaved multi-channel samples to mono by averaging the
// channels of each frame. Mono input is returned unchanged.
func downmix(interleaved []float32, channels int) []float32 {
	if channels == 1 {
		return interleaved
	}
	frames := len(interleaved) / channels
	out := make([]float32, frames)
	for f := range frames {
		var sum float32
		for c := range channels {
			sum += interleaved[f*channels+c]
		}
		out[f] = sum / float32(channels)
	}
	return out
}
