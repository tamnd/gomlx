// SPDX-License-Identifier: Apache-2.0

package doctor

import (
	"strings"
	"testing"
	"time"
)

// advancingClock returns a clock that moves forward by step on every call, so
// each check records a deterministic, non-zero duration.
func advancingClock(step time.Duration) func() time.Time {
	now := time.Unix(0, 0)
	return func() time.Time {
		t := now
		now = now.Add(step)
		return t
	}
}

func TestRunnerTimesAndOrdersChecks(t *testing.T) {
	r := newRunnerClock(advancingClock(time.Millisecond))
	r.Add("a", func() Result { return Pass("first") }).
		Add("b", func() Result { return Fail("second") })
	rep := r.Run()

	if len(rep.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(rep.Results))
	}
	if rep.Results[0].Name != "a" || rep.Results[1].Name != "b" {
		t.Fatalf("checks ran out of order: %q then %q", rep.Results[0].Name, rep.Results[1].Name)
	}
	if rep.Results[0].Duration != time.Millisecond {
		t.Fatalf("duration=%v want 1ms", rep.Results[0].Duration)
	}
}

func TestReportOKOnlyFailsOnFail(t *testing.T) {
	warnOnly := newRunnerClock(advancingClock(0)).
		Add("w", func() Result { return Warn("noted") }).Run()
	if !warnOnly.OK() {
		t.Fatal("a warning should not make the report fail")
	}
	withFail := newRunnerClock(advancingClock(0)).
		Add("w", func() Result { return Warn("noted") }).
		Add("f", func() Result { return Fail("broken") }).Run()
	if withFail.OK() {
		t.Fatal("a failing check should make the report not OK")
	}
}

func TestReportCounts(t *testing.T) {
	rep := newRunnerClock(advancingClock(0)).
		Add("p1", func() Result { return Pass("") }).
		Add("p2", func() Result { return Pass("") }).
		Add("w", func() Result { return Warn("") }).
		Add("s", func() Result { return Skip("") }).Run()
	c := rep.Counts()
	if c[StatusPass] != 2 || c[StatusWarn] != 1 || c[StatusSkip] != 1 || c[StatusFail] != 0 {
		t.Fatalf("counts = %v", c)
	}
}

func TestMarkdownContainsRowsMetricsAndSummary(t *testing.T) {
	rep := newRunnerClock(advancingClock(0)).
		Add("runtime", func() Result {
			return Pass("ok").WithMetrics(map[string]string{"cpus": "8"})
		}).
		Add("models", func() Result { return Warn("none found") }).Run()
	md := rep.Markdown()

	for _, want := range []string{
		"| runtime | pass | ok |",
		"| models | warn | none found |",
		"## Metrics",
		"runtime: cpus=8",
		"Summary: pass=1 warn=1 fail=0 skip=0",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown missing %q\n---\n%s", want, md)
		}
	}
}

func TestMarkdownEscapesPipesAndNewlines(t *testing.T) {
	rep := newRunnerClock(advancingClock(0)).
		Add("c", func() Result { return Fail("a|b\nsecond line") }).Run()
	md := rep.Markdown()
	if strings.Contains(md, "a|b") {
		t.Errorf("unescaped pipe leaked into table:\n%s", md)
	}
	if !strings.Contains(md, "a\\|b second line") {
		t.Errorf("detail not sanitized:\n%s", md)
	}
}

func TestPlatformResult(t *testing.T) {
	if r := platformResult("darwin", "arm64"); r.Status != StatusPass {
		t.Errorf("apple silicon should pass, got %s", r.Status)
	}
	if r := platformResult("linux", "amd64"); r.Status != StatusWarn {
		t.Errorf("non-apple should warn, got %s", r.Status)
	}
}

func TestRuntimeCheckReportsMetrics(t *testing.T) {
	r := RuntimeCheck()
	if r.Status != StatusPass {
		t.Fatalf("runtime check status=%s", r.Status)
	}
	if r.Metrics["go"] == "" || r.Metrics["cpus"] == "" {
		t.Fatalf("missing runtime metrics: %v", r.Metrics)
	}
}
