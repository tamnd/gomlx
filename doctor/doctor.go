// SPDX-License-Identifier: Apache-2.0

// Package doctor runs a series of named checks against the host and the local
// model cache and reports whether the machine is ready to serve. Each check is a
// small function that returns a Result; a Runner executes an ordered list of
// them, times each one, and folds the outcomes into a Report that renders as
// markdown. Keeping the framework this plain means a new check is just a function
// added to the default set, with nothing else to wire up.
package doctor

import (
	"sort"
	"strconv"
	"strings"
	"time"
)

// Status is the outcome of a single check.
type Status string

const (
	// StatusPass means the check found nothing wrong.
	StatusPass Status = "pass"
	// StatusWarn means the check found something worth noting that does not by
	// itself stop the server from running.
	StatusWarn Status = "warn"
	// StatusFail means the check found a problem that will stop the server from
	// working as expected.
	StatusFail Status = "fail"
	// StatusSkip means the check did not run, usually because a precondition was
	// missing.
	StatusSkip Status = "skip"
)

// Result is what a check function returns. The Runner attaches the name and the
// measured duration, so a check only describes its own outcome.
type Result struct {
	Status  Status
	Detail  string
	Metrics map[string]string
}

// Pass returns a passing Result with an optional detail line.
func Pass(detail string) Result { return Result{Status: StatusPass, Detail: detail} }

// Warn returns a warning Result.
func Warn(detail string) Result { return Result{Status: StatusWarn, Detail: detail} }

// Fail returns a failing Result.
func Fail(detail string) Result { return Result{Status: StatusFail, Detail: detail} }

// Skip returns a skipped Result.
func Skip(detail string) Result { return Result{Status: StatusSkip, Detail: detail} }

// WithMetrics returns a copy of r carrying the given metrics, so a check can
// report numbers alongside its status without building the struct by hand.
func (r Result) WithMetrics(m map[string]string) Result {
	r.Metrics = m
	return r
}

// CheckResult is one check's outcome enriched with its name and how long it took.
type CheckResult struct {
	Name     string
	Status   Status
	Detail   string
	Metrics  map[string]string
	Duration time.Duration
}

// Check pairs a name with the function that produces its Result.
type Check struct {
	Name string
	Run  func() Result
}

// Runner holds an ordered list of checks and runs them in turn.
type Runner struct {
	now    func() time.Time
	checks []Check
}

// NewRunner returns an empty Runner using the wall clock.
func NewRunner() *Runner { return newRunnerClock(time.Now) }

func newRunnerClock(now func() time.Time) *Runner {
	return &Runner{now: now}
}

// Add appends a check and returns the Runner so calls can be chained.
func (r *Runner) Add(name string, fn func() Result) *Runner {
	r.checks = append(r.checks, Check{Name: name, Run: fn})
	return r
}

// Run executes every check in order, timing each, and returns the Report.
func (r *Runner) Run() Report {
	rep := Report{GeneratedAt: r.now()}
	for _, c := range r.checks {
		start := r.now()
		res := c.Run()
		rep.Results = append(rep.Results, CheckResult{
			Name:     c.Name,
			Status:   res.Status,
			Detail:   res.Detail,
			Metrics:  res.Metrics,
			Duration: r.now().Sub(start),
		})
	}
	return rep
}

// Report is the result of a full run.
type Report struct {
	GeneratedAt time.Time
	Results     []CheckResult
}

// OK reports whether the run is clean, meaning no check failed. Warnings do not
// make a report fail.
func (rep Report) OK() bool {
	for _, r := range rep.Results {
		if r.Status == StatusFail {
			return false
		}
	}
	return true
}

// Counts tallies how many checks ended in each status.
func (rep Report) Counts() map[Status]int {
	out := map[Status]int{}
	for _, r := range rep.Results {
		out[r.Status]++
	}
	return out
}

// Markdown renders the report as a table followed by a one-line summary. Metrics
// are listed under their check so the table stays narrow.
func (rep Report) Markdown() string {
	var b strings.Builder
	b.WriteString("# Doctor report\n\n")
	b.WriteString("| Check | Status | Detail |\n")
	b.WriteString("| --- | --- | --- |\n")
	for _, r := range rep.Results {
		b.WriteString("| ")
		b.WriteString(mdCell(r.Name))
		b.WriteString(" | ")
		b.WriteString(string(r.Status))
		b.WriteString(" | ")
		b.WriteString(mdCell(r.Detail))
		b.WriteString(" |\n")
	}
	rep.writeMetrics(&b)
	c := rep.Counts()
	b.WriteString("\n")
	b.WriteString("Summary: ")
	b.WriteString(summary(c))
	b.WriteString("\n")
	return b.String()
}

func (rep Report) writeMetrics(b *strings.Builder) {
	first := true
	for _, r := range rep.Results {
		if len(r.Metrics) == 0 {
			continue
		}
		if first {
			b.WriteString("\n## Metrics\n\n")
			first = false
		}
		b.WriteString("- ")
		b.WriteString(r.Name)
		b.WriteString(": ")
		keys := make([]string, 0, len(r.Metrics))
		for k := range r.Metrics {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, k+"="+r.Metrics[k])
		}
		b.WriteString(strings.Join(parts, ", "))
		b.WriteString("\n")
	}
}

// summary renders the per-status counts in a stable order.
func summary(c map[Status]int) string {
	order := []Status{StatusPass, StatusWarn, StatusFail, StatusSkip}
	parts := make([]string, 0, len(order))
	for _, s := range order {
		parts = append(parts, string(s)+"="+strconv.Itoa(c[s]))
	}
	return strings.Join(parts, " ")
}

// mdCell makes a string safe to drop into a markdown table cell: newlines folded
// to spaces and pipes escaped so they do not split the row.
func mdCell(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	if s == "" {
		return "-"
	}
	return s
}
