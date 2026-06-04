// SPDX-License-Identifier: Apache-2.0

// Package vision turns the image bytes carried by a multimodal request into the
// normalized pixel tensor a vision encoder consumes. A request delivers an image
// as raw PNG or JPEG bytes (decoded from a data URL, or fetched from a URL); a
// vision tower expects a fixed-size, channel-first float tensor whose pixels have
// been rescaled and normalized. This package bridges the two with a pure-Go
// decode, resize, and normalize step so the conversion is testable without a GPU.
package vision

import (
	"bytes"
	"fmt"
	"image"
	_ "image/jpeg" // register the JPEG decoder
	_ "image/png"  // register the PNG decoder
)

// PreprocessConfig describes how a vision encoder expects its pixels: the target
// spatial size to resize to, the factor that maps an 8-bit channel value into the
// rescaled range (1/255 for the usual 0..1), and the per-channel mean and
// standard deviation used to normalize each rescaled channel. Mean and Std are in
// red, green, blue order.
type PreprocessConfig struct {
	Width         int
	Height        int
	RescaleFactor float32
	Mean          [3]float32
	Std           [3]float32
}

// DefaultPreprocessConfig is the CLIP-style preprocessing most vision towers use:
// resize to 224 by 224, rescale by 1/255, and normalize with the OpenAI CLIP mean
// and standard deviation.
func DefaultPreprocessConfig() PreprocessConfig {
	return PreprocessConfig{
		Width:         224,
		Height:        224,
		RescaleFactor: 1.0 / 255.0,
		Mean:          [3]float32{0.48145466, 0.4578275, 0.40821073},
		Std:           [3]float32{0.26862954, 0.26130258, 0.27577711},
	}
}

// PixelValues is a normalized image tensor in channel-first (CHW) order: the red
// plane, then green, then blue, each Height by Width and laid out row by row. The
// CHW layout matches what a convolutional patch embedding reads.
type PixelValues struct {
	Channels int
	Height   int
	Width    int
	Data     []float32
}

// At returns the value at channel c, row y, column x. It is a convenience for
// tests and callers that index the tensor without recomputing the stride.
func (p *PixelValues) At(c, y, x int) float32 {
	return p.Data[(c*p.Height+y)*p.Width+x]
}

// Decode decodes PNG or JPEG bytes into an image. The format is detected from the
// data, so a caller does not need to know which encoding the client sent.
func Decode(data []byte) (image.Image, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("vision: empty image data")
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("vision: decode image: %w", err)
	}
	return img, nil
}

// Preprocess decodes image bytes and produces normalized pixel values ready for a
// vision encoder, applying the resize, rescale, and per-channel normalization in
// cfg.
func Preprocess(data []byte, cfg PreprocessConfig) (*PixelValues, error) {
	img, err := Decode(data)
	if err != nil {
		return nil, err
	}
	return PreprocessImage(img, cfg)
}

// PreprocessImage normalizes an already-decoded image. It is split out so a caller
// that decoded the image itself, or built one in memory, can reuse the resize and
// normalize path.
func PreprocessImage(img image.Image, cfg PreprocessConfig) (*PixelValues, error) {
	if cfg.Width <= 0 || cfg.Height <= 0 {
		return nil, fmt.Errorf("vision: target size must be positive, got %dx%d", cfg.Width, cfg.Height)
	}
	for c := range 3 {
		if cfg.Std[c] == 0 {
			return nil, fmt.Errorf("vision: std for channel %d is zero", c)
		}
	}

	resized := resizeBilinear(img, cfg.Width, cfg.Height)

	out := &PixelValues{
		Channels: 3,
		Height:   cfg.Height,
		Width:    cfg.Width,
		Data:     make([]float32, 3*cfg.Height*cfg.Width),
	}
	plane := cfg.Height * cfg.Width
	for y := range cfg.Height {
		for x := range cfg.Width {
			r, g, b := resized[y*cfg.Width+x][0], resized[y*cfg.Width+x][1], resized[y*cfg.Width+x][2]
			channels := [3]float32{r, g, b}
			idx := y*cfg.Width + x
			for c := range 3 {
				v := channels[c]*cfg.RescaleFactor - cfg.Mean[c]
				out.Data[c*plane+idx] = v / cfg.Std[c]
			}
		}
	}
	return out, nil
}

// resizeBilinear resizes the image to dstW by dstH using bilinear interpolation,
// returning the result as a flat row-major slice of [r, g, b] 8-bit values held
// in float32 so the normalize step works in one numeric type. Bilinear is the
// resampling Hugging Face image processors use by default, so this matches the
// reference preprocessing closely enough for encoder parity.
func resizeBilinear(img image.Image, dstW, dstH int) [][3]float32 {
	b := img.Bounds()
	srcW, srcH := b.Dx(), b.Dy()
	out := make([][3]float32, dstW*dstH)

	// A single-pixel source has no interval to interpolate over; map every
	// destination pixel to it. Otherwise sample at destination-pixel centers so
	// the corners line up with the source corners.
	scaleX, scaleY := 0.0, 0.0
	if dstW > 1 {
		scaleX = float64(srcW-1) / float64(dstW-1)
	}
	if dstH > 1 {
		scaleY = float64(srcH-1) / float64(dstH-1)
	}

	for dy := range dstH {
		sy := float64(dy) * scaleY
		y0 := int(sy)
		y1 := min(y0+1, srcH-1)
		wy := float32(sy - float64(y0))
		for dx := range dstW {
			sx := float64(dx) * scaleX
			x0 := int(sx)
			x1 := min(x0+1, srcW-1)
			wx := float32(sx - float64(x0))

			c00 := rgb(img, b.Min.X+x0, b.Min.Y+y0)
			c10 := rgb(img, b.Min.X+x1, b.Min.Y+y0)
			c01 := rgb(img, b.Min.X+x0, b.Min.Y+y1)
			c11 := rgb(img, b.Min.X+x1, b.Min.Y+y1)

			var px [3]float32
			for c := range 3 {
				top := c00[c]*(1-wx) + c10[c]*wx
				bot := c01[c]*(1-wx) + c11[c]*wx
				px[c] = top*(1-wy) + bot*wy
			}
			out[dy*dstW+dx] = px
		}
	}
	return out
}

// rgb reads a pixel as three 8-bit channel values in float32. image.Color reports
// 16-bit pre-multiplied values, so shifting by 8 recovers the 0..255 range the
// rescale factor expects.
func rgb(img image.Image, x, y int) [3]float32 {
	r, g, b, _ := img.At(x, y).RGBA()
	return [3]float32{float32(r >> 8), float32(g >> 8), float32(b >> 8)}
}
