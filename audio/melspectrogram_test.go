// SPDX-License-Identifier: Apache-2.0

package audio

import (
	"math"
	"testing"
)

// tone builds a sine wave of the given frequency, n samples long, at rate hz.
func tone(freq float64, n, rate int) *Waveform {
	s := make([]float32, n)
	for i := range n {
		s[i] = float32(math.Sin(2 * math.Pi * freq * float64(i) / float64(rate)))
	}
	return &Waveform{SampleRate: rate, Samples: s}
}

func TestMelHzRoundTrip(t *testing.T) {
	for _, htk := range []bool{false, true} {
		for _, hz := range []float64{0, 100, 500, 1000, 4000, 8000} {
			got := melToHz(hzToMel(hz, htk), htk)
			if math.Abs(got-hz) > 1e-6 {
				t.Errorf("htk=%v hz=%v round trip got %v", htk, hz, got)
			}
		}
	}
	// The mel scale must be strictly increasing in frequency.
	prev := hzToMel(0, false)
	for hz := 100.0; hz <= 8000; hz += 100 {
		m := hzToMel(hz, false)
		if m <= prev {
			t.Fatalf("mel not increasing at %v: %v after %v", hz, m, prev)
		}
		prev = m
	}
}

func TestMelFilterbankShape(t *testing.T) {
	fb := melFilterbank(80, 400, 16000, false)
	if len(fb) != 80 {
		t.Fatalf("filters: got %d want 80", len(fb))
	}
	nFreqs := 400/2 + 1
	for m, row := range fb {
		if len(row) != nFreqs {
			t.Fatalf("filter %d width: got %d want %d", m, len(row), nFreqs)
		}
		for _, v := range row {
			if v < 0 {
				t.Fatalf("filter %d has a negative weight %v", m, v)
			}
		}
	}
}

func TestLogMelShape(t *testing.T) {
	cfg := DefaultMelConfig()
	w := tone(440, 16000, 16000) // one second
	s, err := w.LogMelSpectrogram(cfg)
	if err != nil {
		t.Fatalf("logmel: %v", err)
	}
	if s.NMels != 80 {
		t.Errorf("mels: got %d want 80", s.NMels)
	}
	wantFrames := 1 + (16000-cfg.NFFT)/cfg.Hop
	if s.NFrames != wantFrames {
		t.Errorf("frames: got %d want %d", s.NFrames, wantFrames)
	}
	if len(s.Data) != s.NMels*s.NFrames {
		t.Errorf("data length: got %d want %d", len(s.Data), s.NMels*s.NFrames)
	}
}

// melBinOfPeak returns the mel bin with the most energy, averaged across frames.
func melBinOfPeak(s *MelSpectrogram) int {
	best, bestVal := 0, float32(math.Inf(-1))
	for m := range s.NMels {
		var sum float32
		for t := range s.NFrames {
			sum += s.At(m, t)
		}
		if sum > bestVal {
			bestVal, best = sum, m
		}
	}
	return best
}

func TestLogMelToneLandsInExpectedBand(t *testing.T) {
	// A low tone must peak in a lower mel bin than a high tone, confirming the
	// frequency axis is wired the right way round.
	cfg := DefaultMelConfig()
	low, err := tone(300, 16000, 16000).LogMelSpectrogram(cfg)
	if err != nil {
		t.Fatalf("low: %v", err)
	}
	high, err := tone(4000, 16000, 16000).LogMelSpectrogram(cfg)
	if err != nil {
		t.Fatalf("high: %v", err)
	}
	lowBin := melBinOfPeak(low)
	highBin := melBinOfPeak(high)
	if lowBin >= highBin {
		t.Errorf("low tone peaked at bin %d, high tone at bin %d; expected low < high", lowBin, highBin)
	}
}

func TestLogMelSilenceIsConstantFloor(t *testing.T) {
	// Silence has uniform (zero) energy, so every value collapses to the same
	// clamped floor.
	w := &Waveform{SampleRate: 16000, Samples: make([]float32, 16000)}
	s, err := w.LogMelSpectrogram(DefaultMelConfig())
	if err != nil {
		t.Fatalf("logmel: %v", err)
	}
	first := s.Data[0]
	for i, v := range s.Data {
		if math.Abs(float64(v-first)) > 1e-5 {
			t.Fatalf("silence value %d = %v differs from %v", i, v, first)
		}
	}
}

func TestLogMelShortInputPadded(t *testing.T) {
	// An input shorter than one FFT frame must still yield one frame, not an error
	// or an empty result.
	w := &Waveform{SampleRate: 16000, Samples: make([]float32, 100)}
	s, err := w.LogMelSpectrogram(DefaultMelConfig())
	if err != nil {
		t.Fatalf("logmel: %v", err)
	}
	if s.NFrames != 1 {
		t.Errorf("short input frames: got %d want 1", s.NFrames)
	}
}

func TestLogMelRejectsBadConfig(t *testing.T) {
	w := tone(440, 1600, 16000)
	if _, err := w.LogMelSpectrogram(MelConfig{NFFT: 0, Hop: 160, NMels: 80}); err == nil {
		t.Error("a zero FFT size must error")
	}
	bad := &Waveform{SampleRate: 0, Samples: []float32{1, 2, 3}}
	if _, err := bad.LogMelSpectrogram(DefaultMelConfig()); err == nil {
		t.Error("a zero sample rate must error")
	}
}
