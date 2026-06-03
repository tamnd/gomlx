// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"sync"
)

// When a model answers with tool calls, something has to run them and feed the
// results back as messages for the next turn. That is the executor. It takes the
// tool calls straight from a model response, runs each against the manager, and
// formats every outcome as a tool-role message tagged with the call id the model
// used, so the conversation can continue. A call that fails for any reason
// becomes an error result rather than a thrown error, because the model is
// expected to read the failure and react, not have the turn aborted under it.

// DefaultMaxParallel is the default ceiling on tool calls run at once.
const DefaultMaxParallel = 4

// toolCaller is the slice of the manager the executor needs. An interface keeps
// the executor testable without a live manager and a set of servers.
type toolCaller interface {
	CallTool(ctx context.Context, fullName string, args map[string]any) (ToolResult, error)
}

// Executor runs the tool calls from a model response against a tool caller.
type Executor struct {
	caller      toolCaller
	maxParallel int
}

// NewExecutor returns an executor that runs calls through caller, with at most
// maxParallel running at once in parallel mode. A non-positive maxParallel falls
// back to the default.
func NewExecutor(caller toolCaller, maxParallel int) *Executor {
	if maxParallel < 1 {
		maxParallel = DefaultMaxParallel
	}
	return &Executor{caller: caller, maxParallel: maxParallel}
}

// CallOutcome pairs a tool result with the id of the call that produced it.
type CallOutcome struct {
	CallID string
	Result ToolResult
}

// Execute runs every tool call and returns the outcomes in the same order as the
// input. With parallel set, calls run concurrently up to the executor's ceiling;
// otherwise they run one at a time. A call that errors yields an error result, so
// every input call has exactly one outcome.
func (e *Executor) Execute(ctx context.Context, toolCalls []map[string]any, parallel bool) []CallOutcome {
	if len(toolCalls) == 0 {
		return nil
	}
	if parallel {
		return e.executeParallel(ctx, toolCalls)
	}
	outcomes := make([]CallOutcome, len(toolCalls))
	for i, tc := range toolCalls {
		outcomes[i] = e.runOne(ctx, tc)
	}
	return outcomes
}

// executeParallel runs the calls concurrently, bounded by maxParallel, keeping the
// outcomes aligned to the input order.
func (e *Executor) executeParallel(ctx context.Context, toolCalls []map[string]any) []CallOutcome {
	outcomes := make([]CallOutcome, len(toolCalls))
	sem := make(chan struct{}, e.maxParallel)
	var wg sync.WaitGroup
	for i, tc := range toolCalls {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, tc map[string]any) {
			defer wg.Done()
			defer func() { <-sem }()
			outcomes[i] = e.runOne(ctx, tc)
		}(i, tc)
	}
	wg.Wait()
	return outcomes
}

// runOne resolves and runs a single tool call. A reachability, routing, or gate
// failure is turned into an error result carrying the call id and tool name, so
// the model sees a tool message rather than the turn ending.
func (e *Executor) runOne(ctx context.Context, toolCall map[string]any) CallOutcome {
	callID, _ := toolCall["id"].(string)
	fn, _ := toolCall["function"].(map[string]any)
	fullName, _ := fn["name"].(string)

	_, _, args := ParseOpenAICall(toolCall)
	result, err := e.caller.CallTool(ctx, fullName, args)
	if err != nil {
		return CallOutcome{
			CallID: callID,
			Result: ToolResult{ToolName: fullName, IsError: true, ErrorMessage: err.Error()},
		}
	}
	return CallOutcome{CallID: callID, Result: result}
}

// ExecuteAndFormat runs the tool calls and renders each outcome as an OpenAI
// tool-role message ready to append to the conversation.
func (e *Executor) ExecuteAndFormat(ctx context.Context, toolCalls []map[string]any, parallel bool) []map[string]any {
	outcomes := e.Execute(ctx, toolCalls, parallel)
	messages := make([]map[string]any, 0, len(outcomes))
	for _, o := range outcomes {
		messages = append(messages, o.Result.ToMessage(o.CallID))
	}
	return messages
}
