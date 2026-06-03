// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// recordingCaller is a fake tool caller that records the calls it received and
// answers from a scripted table.
type recordingCaller struct {
	mu       sync.Mutex
	calls    []string
	results  map[string]ToolResult
	errs     map[string]error
	inFlight atomic.Int32
	maxSeen  atomic.Int32
	delay    time.Duration
}

func (c *recordingCaller) CallTool(ctx context.Context, fullName string, args map[string]any) (ToolResult, error) {
	n := c.inFlight.Add(1)
	for {
		m := c.maxSeen.Load()
		if n <= m || c.maxSeen.CompareAndSwap(m, n) {
			break
		}
	}
	if c.delay > 0 {
		time.Sleep(c.delay)
	}
	c.inFlight.Add(-1)

	c.mu.Lock()
	c.calls = append(c.calls, fullName)
	c.mu.Unlock()

	if err, ok := c.errs[fullName]; ok {
		return ToolResult{}, err
	}
	if r, ok := c.results[fullName]; ok {
		return r, nil
	}
	return ToolResult{ToolName: fullName, Content: "ok"}, nil
}

func toolCall(id, name, args string) map[string]any {
	return map[string]any{
		"id":       id,
		"type":     "function",
		"function": map[string]any{"name": name, "arguments": args},
	}
}

func TestExecutorSequentialFormatsMessages(t *testing.T) {
	caller := &recordingCaller{
		results: map[string]ToolResult{
			"s__a": {ToolName: "s__a", Content: "result a"},
		},
	}
	e := NewExecutor(caller, 0) // default ceiling

	msgs := e.ExecuteAndFormat(context.Background(), []map[string]any{
		toolCall("call-1", "s__a", `{"x":1}`),
	}, false)

	if len(msgs) != 1 {
		t.Fatalf("want 1 message, got %d", len(msgs))
	}
	m := msgs[0]
	if m["role"] != "tool" || m["tool_call_id"] != "call-1" || m["content"] != "result a" {
		t.Fatalf("tool message wrong: %v", m)
	}
}

func TestExecutorPreservesOrder(t *testing.T) {
	caller := &recordingCaller{}
	e := NewExecutor(caller, 8)

	var calls []map[string]any
	for i := 0; i < 6; i++ {
		calls = append(calls, toolCall(fmt.Sprintf("c%d", i), fmt.Sprintf("s__t%d", i), "{}"))
	}
	outcomes := e.Execute(context.Background(), calls, true)
	if len(outcomes) != 6 {
		t.Fatalf("want 6 outcomes, got %d", len(outcomes))
	}
	// Outcomes must line up with input order regardless of completion order.
	for i, o := range outcomes {
		if o.CallID != fmt.Sprintf("c%d", i) {
			t.Fatalf("outcome %d has call id %q", i, o.CallID)
		}
	}
}

func TestExecutorErrorBecomesErrorResult(t *testing.T) {
	caller := &recordingCaller{
		errs: map[string]error{"s__bad": errors.New("server refused")},
	}
	e := NewExecutor(caller, 1)

	outcomes := e.Execute(context.Background(), []map[string]any{
		toolCall("c1", "s__bad", "{}"),
	}, false)
	if len(outcomes) != 1 {
		t.Fatalf("want 1 outcome, got %d", len(outcomes))
	}
	res := outcomes[0].Result
	if !res.IsError || res.ErrorMessage != "server refused" {
		t.Fatalf("error not surfaced as an error result: %+v", res)
	}
	// And it still formats as a tool message, so the conversation can continue.
	msg := res.ToMessage("c1")
	if msg["content"] != "Error: server refused" {
		t.Fatalf("error message content wrong: %v", msg["content"])
	}
}

func TestExecutorRespectsParallelCeiling(t *testing.T) {
	caller := &recordingCaller{delay: 30 * time.Millisecond}
	e := NewExecutor(caller, 2) // at most two at once

	var calls []map[string]any
	for i := 0; i < 6; i++ {
		calls = append(calls, toolCall(fmt.Sprintf("c%d", i), fmt.Sprintf("s__t%d", i), "{}"))
	}
	e.Execute(context.Background(), calls, true)

	if got := caller.maxSeen.Load(); got > 2 {
		t.Fatalf("max concurrent calls=%d, ceiling was 2", got)
	}
}

func TestExecutorSequentialRunsOneAtATime(t *testing.T) {
	caller := &recordingCaller{delay: 10 * time.Millisecond}
	e := NewExecutor(caller, 8)

	var calls []map[string]any
	for i := 0; i < 4; i++ {
		calls = append(calls, toolCall(fmt.Sprintf("c%d", i), fmt.Sprintf("s__t%d", i), "{}"))
	}
	e.Execute(context.Background(), calls, false)

	if got := caller.maxSeen.Load(); got != 1 {
		t.Fatalf("sequential execution saw %d concurrent calls, want 1", got)
	}
}

func TestExecutorEmptyInput(t *testing.T) {
	e := NewExecutor(&recordingCaller{}, 4)
	if got := e.Execute(context.Background(), nil, true); got != nil {
		t.Fatalf("no tool calls should yield nil, got %v", got)
	}
}

func TestExecutorPassesParsedArguments(t *testing.T) {
	var gotArgs map[string]any
	caller := callerFunc(func(ctx context.Context, name string, args map[string]any) (ToolResult, error) {
		gotArgs = args
		return ToolResult{ToolName: name, Content: "ok"}, nil
	})
	e := NewExecutor(caller, 1)
	e.Execute(context.Background(), []map[string]any{
		toolCall("c1", "s__a", `{"path":"x.txt","n":2}`),
	}, false)
	if gotArgs["path"] != "x.txt" {
		t.Fatalf("arguments not parsed and passed through: %v", gotArgs)
	}
}

// callerFunc adapts a function to the toolCaller interface.
type callerFunc func(ctx context.Context, fullName string, args map[string]any) (ToolResult, error)

func (f callerFunc) CallTool(ctx context.Context, fullName string, args map[string]any) (ToolResult, error) {
	return f(ctx, fullName, args)
}
