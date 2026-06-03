// SPDX-License-Identifier: Apache-2.0

package doctor

import (
	"runtime"
	"strconv"
)

// RuntimeCheck reports the Go runtime the binary is running on. It always passes;
// the version and CPU count are surfaced as metrics for the report.
func RuntimeCheck() Result {
	return Pass("Go runtime detected").WithMetrics(map[string]string{
		"go":      runtime.Version(),
		"cpus":    strconv.Itoa(runtime.NumCPU()),
		"gomaxes": strconv.Itoa(runtime.GOMAXPROCS(0)),
	})
}

// PlatformCheck reports whether the host can run the Metal compute backend. The
// backend needs Apple Silicon, so anything other than darwin/arm64 is a warning
// rather than a hard failure: the pure-Go serving layer still works for
// development, it just cannot run real inference.
func PlatformCheck() Result {
	return platformResult(runtime.GOOS, runtime.GOARCH)
}

func platformResult(goos, goarch string) Result {
	m := map[string]string{"os": goos, "arch": goarch}
	if goos == "darwin" && goarch == "arm64" {
		return Pass("Apple Silicon, Metal backend available").WithMetrics(m)
	}
	return Warn("not Apple Silicon, the Metal backend is unavailable on this host").WithMetrics(m)
}

// ModelsCheck reports how many known aliases have weights in the given cache
// roots. It warns rather than fails when none are present, since a fresh install
// has no models yet and the user is expected to fetch what they want.
func ModelsCheck(roots []string) Result {
	available := AvailableAliases(roots)
	m := map[string]string{
		"roots":     strconv.Itoa(len(roots)),
		"available": strconv.Itoa(len(available)),
	}
	if len(available) == 0 {
		return Warn("no model weights found in any cache root, fetch a model before serving").WithMetrics(m)
	}
	detail := strconv.Itoa(len(available)) + " model(s) ready locally"
	return Pass(detail).WithMetrics(m)
}

// Diagnose runs the default set of checks against the live host and cache and
// returns the report. It is the entry point a CLI command would call.
func Diagnose() Report {
	roots := DefaultCacheRoots()
	return NewRunner().
		Add("runtime", RuntimeCheck).
		Add("platform", PlatformCheck).
		Add("models", func() Result { return ModelsCheck(roots) }).
		Run()
}
