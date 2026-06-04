// SPDX-License-Identifier: Apache-2.0

package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

// encodePCM16 builds a minimal RIFF/WAVE clip from interleaved 16-bit samples,
// enough to exercise the decoder.
func encodePCM16(t *testing.T, channels, rate int, samples []int16) []byte {
	t.Helper()
	var b bytes.Buffer
	dataLen := len(samples) * 2
	byteRate := rate * channels * 2
	blockAlign := channels * 2

	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataLen))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(formatPCM))
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(rate))
	binary.Write(&b, binary.LittleEndian, uint32(byteRate))
	binary.Write(&b, binary.LittleEndian, uint16(blockAlign))
	binary.Write(&b, binary.LittleEndian, uint16(16))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(dataLen))
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}

func encodeFloat32(t *testing.T, channels, rate int, samples []float32) []byte {
	t.Helper()
	var b bytes.Buffer
	dataLen := len(samples) * 4
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+dataLen))
	b.WriteString("WAVE")
	b.WriteString("fmt ")
	binary.Write(&b, binary.LittleEndian, uint32(16))
	binary.Write(&b, binary.LittleEndian, uint16(formatIEEEFloat))
	binary.Write(&b, binary.LittleEndian, uint16(channels))
	binary.Write(&b, binary.LittleEndian, uint32(rate))
	binary.Write(&b, binary.LittleEndian, uint32(rate*channels*4))
	binary.Write(&b, binary.LittleEndian, uint16(channels*4))
	binary.Write(&b, binary.LittleEndian, uint16(32))
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(dataLen))
	for _, s := range samples {
		binary.Write(&b, binary.LittleEndian, s)
	}
	return b.Bytes()
}

func TestDecodeWAVPCM16(t *testing.T) {
	data := encodePCM16(t, 1, 16000, []int16{0, 16384, -16384, 32767, -32768})
	w, err := DecodeWAV(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if w.SampleRate != 16000 {
		t.Errorf("rate: got %d want 16000", w.SampleRate)
	}
	want := []float32{0, 0.5, -0.5, 32767.0 / 32768.0, -1.0}
	if len(w.Samples) != len(want) {
		t.Fatalf("samples: got %d want %d", len(w.Samples), len(want))
	}
	for i := range want {
		if math.Abs(float64(w.Samples[i]-want[i])) > 1e-4 {
			t.Errorf("sample %d: got %v want %v", i, w.Samples[i], want[i])
		}
	}
}

func TestDecodeWAVStereoDownmix(t *testing.T) {
	// Two frames of stereo: frame 0 = (1.0, 0.0) -> 0.5; frame 1 = (-1.0, -1.0) -> -1.0.
	data := encodePCM16(t, 2, 16000, []int16{32767, 0, -32768, -32768})
	w, err := DecodeWAV(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(w.Samples) != 2 {
		t.Fatalf("frames: got %d want 2", len(w.Samples))
	}
	if math.Abs(float64(w.Samples[0]-0.5)) > 1e-3 {
		t.Errorf("frame 0 downmix: got %v want ~0.5", w.Samples[0])
	}
	if math.Abs(float64(w.Samples[1]+1.0)) > 1e-3 {
		t.Errorf("frame 1 downmix: got %v want ~-1.0", w.Samples[1])
	}
}

func TestDecodeWAVFloat32(t *testing.T) {
	data := encodeFloat32(t, 1, 8000, []float32{0.25, -0.75, 1.0})
	w, err := DecodeWAV(data)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	want := []float32{0.25, -0.75, 1.0}
	for i := range want {
		if math.Abs(float64(w.Samples[i]-want[i])) > 1e-6 {
			t.Errorf("sample %d: got %v want %v", i, w.Samples[i], want[i])
		}
	}
}

func TestDecodeWAVRejectsBad(t *testing.T) {
	if _, err := DecodeWAV([]byte("not a wav")); err == nil {
		t.Error("non-RIFF data must error")
	}
	if _, err := DecodeWAV(nil); err == nil {
		t.Error("empty data must error")
	}
	// 24-bit PCM is unsupported and must be reported, not silently mangled.
	data := encodePCM16(t, 1, 16000, []int16{1, 2})
	// Corrupt the bits-per-sample field to 24.
	idx := bytes.Index(data, []byte("fmt ")) + 8 + 14
	binary.LittleEndian.PutUint16(data[idx:idx+2], 24)
	if _, err := DecodeWAV(data); err == nil {
		t.Error("24-bit PCM must be rejected")
	}
}

func TestResampleDownAndUp(t *testing.T) {
	w := &Waveform{SampleRate: 16000, Samples: make([]float32, 16000)}
	for i := range w.Samples {
		w.Samples[i] = float32(i)
	}
	down, err := w.Resample(8000)
	if err != nil {
		t.Fatalf("resample: %v", err)
	}
	if down.SampleRate != 8000 {
		t.Errorf("rate: got %d want 8000", down.SampleRate)
	}
	if len(down.Samples) != 8000 {
		t.Errorf("length: got %d want 8000", len(down.Samples))
	}
	// Halving the rate samples every other point of a ramp: out[i] ~ 2*i.
	if math.Abs(float64(down.Samples[100]-200)) > 1.0 {
		t.Errorf("downsampled ramp at 100: got %v want ~200", down.Samples[100])
	}
}

func TestResampleNoOp(t *testing.T) {
	w := &Waveform{SampleRate: 16000, Samples: []float32{0.1, 0.2, 0.3}}
	got, err := w.Resample(16000)
	if err != nil {
		t.Fatalf("resample: %v", err)
	}
	if len(got.Samples) != 3 || got.Samples[1] != 0.2 {
		t.Errorf("same-rate resample should pass samples through, got %v", got.Samples)
	}
}

func TestResampleRejectsBadTarget(t *testing.T) {
	w := &Waveform{SampleRate: 16000, Samples: []float32{1, 2, 3}}
	if _, err := w.Resample(0); err == nil {
		t.Error("a zero target rate must error")
	}
}

func TestDecodeForSpeech(t *testing.T) {
	// An 8 kHz clip must come back at the 16 kHz speech rate.
	data := encodePCM16(t, 1, 8000, []int16{0, 8192, 16384, 24576, 32767, 0, -16384, -32768})
	w, err := DecodeForSpeech(data)
	if err != nil {
		t.Fatalf("decode for speech: %v", err)
	}
	if w.SampleRate != SpeechSampleRate {
		t.Errorf("rate: got %d want %d", w.SampleRate, SpeechSampleRate)
	}
	if len(w.Samples) != 16 {
		t.Errorf("length: got %d want 16 (8 samples upsampled 2x)", len(w.Samples))
	}
}
