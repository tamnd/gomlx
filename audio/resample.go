// SPDX-License-Identifier: Apache-2.0

package audio

import "fmt"

// SpeechSampleRate is the rate speech models expect, 16 kHz. A clip recorded at
// any other rate is resampled to this before feature extraction.
const SpeechSampleRate = 16000

// Resample returns a waveform at the target sample rate, interpolating linearly
// between the source samples. A waveform already at the target rate is returned
// unchanged. Linear interpolation is enough for the downsampling speech input
// usually needs; it is not a substitute for a high-quality anti-aliasing
// resampler when upsampling far past the source rate.
func (w *Waveform) Resample(target int) (*Waveform, error) {
	if target <= 0 {
		return nil, fmt.Errorf("audio: target sample rate must be positive, got %d", target)
	}
	if w.SampleRate == target || len(w.Samples) == 0 {
		return &Waveform{SampleRate: target, Samples: w.Samples}, nil
	}
	if w.SampleRate <= 0 {
		return nil, fmt.Errorf("audio: source sample rate must be positive, got %d", w.SampleRate)
	}

	ratio := float64(target) / float64(w.SampleRate)
	n := max(int(float64(len(w.Samples))*ratio), 1)
	out := make([]float32, n)

	// Map each output sample back to a fractional source position and interpolate
	// between its two neighbors. step is the source advance per output sample.
	step := float64(w.SampleRate) / float64(target)
	last := len(w.Samples) - 1
	for i := range n {
		src := float64(i) * step
		i0 := int(src)
		if i0 >= last {
			out[i] = w.Samples[last]
			continue
		}
		frac := float32(src - float64(i0))
		out[i] = w.Samples[i0]*(1-frac) + w.Samples[i0+1]*frac
	}
	return &Waveform{SampleRate: target, Samples: out}, nil
}

// DecodeForSpeech decodes a WAV clip and resamples it to the speech sample rate,
// the one step a transcription front end needs before feature extraction.
func DecodeForSpeech(data []byte) (*Waveform, error) {
	w, err := DecodeWAV(data)
	if err != nil {
		return nil, err
	}
	return w.Resample(SpeechSampleRate)
}
