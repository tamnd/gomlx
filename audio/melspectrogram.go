// SPDX-License-Identifier: Apache-2.0

package audio

import (
	"fmt"
	"math"
)

// A speech encoder does not read a raw waveform: it reads a log-mel spectrogram,
// the waveform's energy laid out over time and a mel-spaced frequency axis. The
// front end frames the signal, takes a windowed Fourier transform of each frame,
// projects the power spectrum onto a bank of mel filters, and takes the log. This
// is the standard Whisper-style feature extraction, done here on the host so the
// waveform-to-features step is testable without the model.

// MelConfig describes the spectrogram: the FFT size, the hop between frames, the
// number of mel filters, and whether to use the HTK mel scale instead of the
// Slaney scale. The defaults are Whisper's.
type MelConfig struct {
	NFFT  int
	Hop   int
	NMels int
	HTK   bool
}

// DefaultMelConfig is Whisper's feature configuration at 16 kHz: a 400-sample FFT
// (25 ms), a 160-sample hop (10 ms), 80 mel filters, and the Slaney mel scale.
func DefaultMelConfig() MelConfig {
	return MelConfig{NFFT: 400, Hop: 160, NMels: 80, HTK: false}
}

// MelSpectrogram is the log-mel feature tensor: NMels rows over NFrames columns,
// stored row-major (all of mel bin 0 across time, then bin 1, ...).
type MelSpectrogram struct {
	NMels   int
	NFrames int
	Data    []float32
}

// At returns the value at mel bin m and frame t.
func (s *MelSpectrogram) At(m, t int) float32 {
	return s.Data[m*s.NFrames+t]
}

// LogMelSpectrogram computes the log-mel features of the waveform. The signal is
// zero-padded to at least one FFT frame, framed with the configured hop, windowed
// with a Hann window, and transformed; each frame's power spectrum is projected
// onto the mel filterbank, and the result is log-scaled with Whisper's
// normalization (log10, clamped to within 8 decades of the peak, then mapped to
// roughly [-1, 1]).
func (w *Waveform) LogMelSpectrogram(cfg MelConfig) (*MelSpectrogram, error) {
	if cfg.NFFT <= 0 || cfg.Hop <= 0 || cfg.NMels <= 0 {
		return nil, fmt.Errorf("audio: mel config must be positive, got nfft=%d hop=%d nmels=%d",
			cfg.NFFT, cfg.Hop, cfg.NMels)
	}
	if w.SampleRate <= 0 {
		return nil, fmt.Errorf("audio: sample rate must be positive, got %d", w.SampleRate)
	}

	samples := w.Samples
	if len(samples) < cfg.NFFT {
		padded := make([]float32, cfg.NFFT)
		copy(padded, samples)
		samples = padded
	}

	nFreqs := cfg.NFFT/2 + 1
	nFrames := 1 + (len(samples)-cfg.NFFT)/cfg.Hop
	window := hannWindow(cfg.NFFT)
	fb := melFilterbank(cfg.NMels, cfg.NFFT, w.SampleRate, cfg.HTK)

	out := &MelSpectrogram{
		NMels:   cfg.NMels,
		NFrames: nFrames,
		Data:    make([]float32, cfg.NMels*nFrames),
	}

	power := make([]float64, nFreqs)
	frame := make([]float64, cfg.NFFT)
	var maxLog float64 = math.Inf(-1)

	for t := range nFrames {
		start := t * cfg.Hop
		for n := range cfg.NFFT {
			frame[n] = float64(samples[start+n]) * window[n]
		}
		powerSpectrum(frame, power)
		for m := range cfg.NMels {
			var energy float64
			row := fb[m]
			for k := range nFreqs {
				energy += row[k] * power[k]
			}
			logv := math.Log10(math.Max(energy, 1e-10))
			if logv > maxLog {
				maxLog = logv
			}
			out.Data[m*nFrames+t] = float32(logv)
		}
	}

	// Whisper clamps each value to within 8 decades of the loudest, then maps the
	// result to roughly [-1, 1].
	floor := maxLog - 8.0
	for i, v := range out.Data {
		lv := math.Max(float64(v), floor)
		out.Data[i] = float32((lv + 4.0) / 4.0)
	}
	return out, nil
}

// powerSpectrum fills power with the squared magnitude of the real DFT of frame
// for bins 0..len(frame)/2. The frame is small (a few hundred samples) so a
// direct transform is clear and fast enough.
func powerSpectrum(frame []float64, power []float64) {
	n := len(frame)
	for k := range power {
		var re, im float64
		angStep := -2.0 * math.Pi * float64(k) / float64(n)
		for t := range n {
			ang := angStep * float64(t)
			re += frame[t] * math.Cos(ang)
			im += frame[t] * math.Sin(ang)
		}
		power[k] = re*re + im*im
	}
}

// hannWindow returns an n-point periodic Hann window, the window Whisper applies
// before the transform.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range n {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n)))
	}
	return w
}

// melFilterbank builds the nMels by (nFFT/2+1) triangular mel filter matrix for
// the given sample rate, Slaney-normalized so each filter integrates to the same
// area. It is the projection from linear-frequency power bins onto mel bins.
func melFilterbank(nMels, nFFT, sampleRate int, htk bool) [][]float64 {
	nFreqs := nFFT/2 + 1
	fftFreqs := make([]float64, nFreqs)
	for k := range nFreqs {
		fftFreqs[k] = float64(k) * float64(sampleRate) / float64(nFFT)
	}

	melMin := hzToMel(0, htk)
	melMax := hzToMel(float64(sampleRate)/2, htk)
	melPoints := make([]float64, nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + (melMax-melMin)*float64(i)/float64(nMels+1)
	}
	hzPoints := make([]float64, len(melPoints))
	for i, m := range melPoints {
		hzPoints[i] = melToHz(m, htk)
	}

	fb := make([][]float64, nMels)
	for m := range nMels {
		fb[m] = make([]float64, nFreqs)
		lower, center, upper := hzPoints[m], hzPoints[m+1], hzPoints[m+2]
		enorm := 2.0 / (hzPoints[m+2] - hzPoints[m])
		for k := range nFreqs {
			f := fftFreqs[k]
			var weight float64
			switch {
			case f >= lower && f <= center && center > lower:
				weight = (f - lower) / (center - lower)
			case f > center && f <= upper && upper > center:
				weight = (upper - f) / (upper - center)
			}
			fb[m][k] = weight * enorm
		}
	}
	return fb
}

// hzToMel converts a frequency in hertz to the mel scale, using either the HTK
// formula or the Slaney scale (linear below 1 kHz, logarithmic above).
func hzToMel(hz float64, htk bool) float64 {
	if htk {
		return 2595.0 * math.Log10(1.0+hz/700.0)
	}
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27.0
	if hz < minLogHz {
		return hz / fSp
	}
	return minLogMel + math.Log(hz/minLogHz)/logStep
}

// melToHz is the inverse of hzToMel.
func melToHz(mel float64, htk bool) float64 {
	if htk {
		return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
	}
	const fSp = 200.0 / 3.0
	const minLogHz = 1000.0
	minLogMel := minLogHz / fSp
	logStep := math.Log(6.4) / 27.0
	if mel < minLogMel {
		return mel * fSp
	}
	return minLogHz * math.Exp(logStep*(mel-minLogMel))
}
