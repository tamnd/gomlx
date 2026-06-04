// SPDX-License-Identifier: Apache-2.0

package vision

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"math"
	"testing"
)

// solidPNG encodes a w by h image filled with one color as PNG bytes.
func solidPNG(t *testing.T, w, h int, c color.Color) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("encode png: %v", err)
	}
	return buf.Bytes()
}

func TestPreprocessShape(t *testing.T) {
	data := solidPNG(t, 64, 48, color.RGBA{10, 20, 30, 255})
	cfg := PreprocessConfig{Width: 16, Height: 12, RescaleFactor: 1.0 / 255.0,
		Mean: [3]float32{0, 0, 0}, Std: [3]float32{1, 1, 1}}

	pv, err := Preprocess(data, cfg)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}
	if pv.Channels != 3 || pv.Height != 12 || pv.Width != 16 {
		t.Fatalf("shape: got %dx%dx%d", pv.Channels, pv.Height, pv.Width)
	}
	if len(pv.Data) != 3*12*16 {
		t.Fatalf("data length: got %d want %d", len(pv.Data), 3*12*16)
	}
}

func TestPreprocessNormalizesSolidColor(t *testing.T) {
	// A solid image resizes to the same color everywhere, so each channel must
	// equal (value/255 - mean) / std exactly.
	const r, g, b = 128, 64, 200
	data := solidPNG(t, 32, 32, color.RGBA{r, g, b, 255})
	cfg := DefaultPreprocessConfig()

	pv, err := Preprocess(data, cfg)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	want := [3]float32{
		(float32(r)*cfg.RescaleFactor - cfg.Mean[0]) / cfg.Std[0],
		(float32(g)*cfg.RescaleFactor - cfg.Mean[1]) / cfg.Std[1],
		(float32(b)*cfg.RescaleFactor - cfg.Mean[2]) / cfg.Std[2],
	}
	for c := range 3 {
		for _, yx := range [][2]int{{0, 0}, {5, 7}, {cfg.Height - 1, cfg.Width - 1}} {
			got := pv.At(c, yx[0], yx[1])
			if math.Abs(float64(got-want[c])) > 1e-4 {
				t.Errorf("channel %d at (%d,%d): got %.5f want %.5f", c, yx[0], yx[1], got, want[c])
			}
		}
	}
}

func TestPreprocessChannelOrderIsRGB(t *testing.T) {
	// Pure red must light up channel 0 and leave channels 1 and 2 at their
	// zero-rescaled-minus-mean value, proving CHW order is red, green, blue.
	data := solidPNG(t, 8, 8, color.RGBA{255, 0, 0, 255})
	cfg := DefaultPreprocessConfig()
	pv, err := Preprocess(data, cfg)
	if err != nil {
		t.Fatalf("preprocess: %v", err)
	}

	red := pv.At(0, 0, 0)
	green := pv.At(1, 0, 0)
	if red <= green {
		t.Errorf("red channel %.4f should exceed green channel %.4f for a red image", red, green)
	}
	wantGreen := (0 - cfg.Mean[1]) / cfg.Std[1]
	if math.Abs(float64(green-wantGreen)) > 1e-4 {
		t.Errorf("green channel: got %.4f want %.4f", green, wantGreen)
	}
}

func TestPreprocessDecodesJPEG(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := range 16 {
		for x := range 16 {
			img.Set(x, y, color.RGBA{100, 100, 100, 255})
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}

	pv, err := Preprocess(buf.Bytes(), PreprocessConfig{Width: 8, Height: 8,
		RescaleFactor: 1.0 / 255.0, Mean: [3]float32{0, 0, 0}, Std: [3]float32{1, 1, 1}})
	if err != nil {
		t.Fatalf("preprocess jpeg: %v", err)
	}
	if pv.Width != 8 || pv.Height != 8 {
		t.Fatalf("jpeg shape: got %dx%d", pv.Width, pv.Height)
	}
}

func TestResizeBilinearGradient(t *testing.T) {
	// A horizontal gradient downsized by 2x should still increase left to right,
	// confirming the interpolation samples across the source rather than picking
	// one pixel.
	src := image.NewRGBA(image.Rect(0, 0, 8, 1))
	for x := range 8 {
		v := uint8(x * 32)
		src.Set(x, 0, color.RGBA{v, v, v, 255})
	}
	out := resizeBilinear(src, 4, 1)
	for i := 1; i < 4; i++ {
		if out[i][0] <= out[i-1][0] {
			t.Errorf("gradient not monotonic at %d: %.1f then %.1f", i, out[i-1][0], out[i][0])
		}
	}
}

func TestPreprocessRejectsBadInput(t *testing.T) {
	if _, err := Preprocess(nil, DefaultPreprocessConfig()); err == nil {
		t.Error("empty data must error")
	}
	if _, err := Preprocess([]byte("not an image"), DefaultPreprocessConfig()); err == nil {
		t.Error("undecodable data must error")
	}
	good := solidPNG(t, 8, 8, color.RGBA{1, 2, 3, 255})
	if _, err := Preprocess(good, PreprocessConfig{Width: 0, Height: 8, Std: [3]float32{1, 1, 1}}); err == nil {
		t.Error("zero width must error")
	}
	if _, err := Preprocess(good, PreprocessConfig{Width: 8, Height: 8, Std: [3]float32{1, 0, 1}}); err == nil {
		t.Error("zero std must error")
	}
}
