// SPDX-License-Identifier: Apache-2.0

// Package engine holds the core request, sampling, and engine types for the
// continuous-batching inference loop.
package engine

import "time"

// RequestStatus is the lifecycle state of a request in the scheduler.
type RequestStatus int

const (
	// StatusWaiting means the request is queued, not yet scheduled.
	StatusWaiting RequestStatus = iota
	// StatusRunning means the request is generating tokens.
	StatusRunning
	// StatusPreempted means the request was evicted and needs resuming.
	StatusPreempted
	// StatusFinishedStopped means generation hit a stop token or string.
	StatusFinishedStopped
	// StatusFinishedLengthCapped means generation hit max_tokens.
	StatusFinishedLengthCapped
	// StatusFinishedAborted means the request was cancelled by the caller.
	StatusFinishedAborted
)

// IsFinished reports whether the status is terminal.
func (s RequestStatus) IsFinished() bool { return s > StatusPreempted }

// FinishReason returns the OpenAI finish_reason string, or "" if not finished.
func (s RequestStatus) FinishReason() string {
	switch s {
	case StatusFinishedStopped:
		return "stop"
	case StatusFinishedLengthCapped:
		return "length"
	case StatusFinishedAborted:
		return "abort"
	default:
		return ""
	}
}

// SamplingParams controls token generation. Defaults mirror the reference.
type SamplingParams struct {
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"` // 0 means disabled
	MinP        float64 `json:"min_p"`

	// RepetitionPenalty is the legacy multiplicative variant (1.0 = disabled).
	RepetitionPenalty float64 `json:"repetition_penalty"`
	// PresencePenalty and FrequencyPenalty are the additive OpenAI variants
	// (0.0 = disabled).
	PresencePenalty  float64 `json:"presence_penalty"`
	FrequencyPenalty float64 `json:"frequency_penalty"`

	Stop         []string `json:"stop,omitempty"`
	StopTokenIDs []int    `json:"stop_token_ids,omitempty"`
}

// DefaultSamplingParams returns the reference defaults.
func DefaultSamplingParams() SamplingParams {
	return SamplingParams{
		MaxTokens:         256,
		Temperature:       0.7,
		TopP:              0.9,
		TopK:              0,
		MinP:              0.0,
		RepetitionPenalty: 1.0,
		PresencePenalty:   0.0,
		FrequencyPenalty:  0.0,
		Stop:              []string{},
		StopTokenIDs:      []int{},
	}
}

// Request is a single inference request tracked by the scheduler, kept simple
// for the MLX backend.
type Request struct {
	RequestID      string
	Prompt         string
	SamplingParams SamplingParams
	ArrivalTime    time.Time
	Priority       int // lower is higher priority

	// Set after tokenization.
	PromptTokenIDs  []int
	NumPromptTokens int

	// Generation state.
	Status            RequestStatus
	NumComputedTokens int
	OutputTokenIDs    []int
	OutputText        string

	// BatchGenerator integration.
	BatchUID *int

	// Prefix-cache fields.
	PromptCache     []any
	CachedTokens    int
	RemainingTokens []int
	PrefixBoundary  int
}
