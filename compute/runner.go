// SPDX-License-Identifier: Apache-2.0

package compute

import (
	"math/rand"
	"sync"

	"github.com/tamnd/gomlx/tokenizer"
)

// Runner decodes many requests at once. A single goroutine owns the device and
// runs one batched forward pass per step over all active sequences, so the
// large projection and MLP matmuls are shared across requests. This is what
// lifts throughput under load above serialized single-stream decode, where each
// request would wait for the one ahead of it.
//
// Batches are formed greedily: the loop takes the first queued request, drains
// up to MaxBatch more without blocking, prefills them together, and decodes
// them in lockstep until every sequence finishes. Requests that arrive while a
// batch is running are served by the next batch.
type Runner struct {
	Model    *Qwen3Model
	Tok      *tokenizer.Tokenizer
	EOS      []int
	MaxBatch int

	jobs chan *job
	eos  map[int]bool
	once sync.Once
}

type job struct {
	promptText string
	prompt     []int // filled by the loop goroutine, where tokenizing is single-threaded
	cfg        GenConfig
	done       chan jobResult
}

type jobResult struct {
	res GenResult
	err error
}

// seqState tracks one sequence's decoding progress within a batch.
type seqState struct {
	cfg       GenConfig
	hist      []int // prompt followed by generated tokens, for the logits processor
	generated []int
	prevText  string
	rng       *rand.Rand
	maxTokens int
	done      *job
	finished  bool
	reason    string
	text      string
}

// Start launches the device loop. It is idempotent and safe to call from
// Generate, so callers need not sequence startup themselves.
func (r *Runner) Start() {
	r.once.Do(func() {
		if r.MaxBatch <= 0 {
			r.MaxBatch = 8
		}
		r.jobs = make(chan *job, 256)
		r.eos = make(map[int]bool, len(r.EOS))
		for _, e := range r.EOS {
			r.eos[e] = true
		}
		go r.loop()
	})
}

// Generate enqueues a request and blocks until it finishes. The signature
// mirrors the single-stream Generator so the engine can swap one for the other.
func (r *Runner) Generate(prompt string, cfg GenConfig) (GenResult, error) {
	r.Start()
	j := &job{promptText: prompt, cfg: cfg, done: make(chan jobResult, 1)}
	r.jobs <- j
	out := <-j.done
	return out.res, out.err
}

func (r *Runner) loop() {
	for {
		batch := r.collect()
		r.run(batch)
	}
}

// collect blocks for the first request, then drains the queue without blocking
// up to the batch limit.
func (r *Runner) collect() []*job {
	first := <-r.jobs
	batch := []*job{first}
	for len(batch) < r.MaxBatch {
		select {
		case j := <-r.jobs:
			batch = append(batch, j)
		default:
			return batch
		}
	}
	return batch
}

// run prefills and decodes a batch to completion, then delivers each result.
func (r *Runner) run(batch []*job) {
	n := len(batch)
	vocab := r.Model.Args.VocabSize

	prompts := make([][]int32, n)
	states := make([]*seqState, n)
	maxSteps := 0
	for i, j := range batch {
		j.prompt = r.Tok.Encode(j.promptText)
		prompts[i] = toInt32(j.prompt)
		mt := j.cfg.MaxTokens
		if mt <= 0 {
			mt = 256
		}
		if mt > maxSteps {
			maxSteps = mt
		}
		states[i] = &seqState{
			cfg:       j.cfg,
			hist:      append([]int(nil), j.prompt...),
			generated: make([]int, 0, mt),
			rng:       rand.New(rand.NewSource(j.cfg.Seed)),
			maxTokens: mt,
			done:      j,
		}
	}

	b := r.Model.NewBatch(n)
	logits, err := b.Prefill(prompts)
	if err != nil {
		for _, j := range batch {
			j.done <- jobResult{err: err}
		}
		return
	}

	for step := 0; step < maxSteps; step++ {
		next := make([]int32, n)
		allDone := true
		for i, st := range states {
			if st.finished {
				continue
			}
			tok := r.stepSeq(st, logits[i*vocab:(i+1)*vocab])
			if st.finished {
				continue
			}
			allDone = false
			next[i] = int32(tok)
		}
		if allDone || step == maxSteps-1 {
			break
		}
		if logits, err = b.Decode(next); err != nil {
			break
		}
	}

	for _, st := range states {
		if !st.finished && err != nil {
			st.done.done <- jobResult{err: err}
			continue
		}
		st.deliver(r.Tok)
	}
}

// stepSeq samples one token for a sequence from its logits row and updates the
// sequence state: it appends the token, emits the decoded delta, and marks the
// sequence finished on an end token, a stop string, or the token budget. It
// returns the sampled token; callers check st.finished to know whether the
// sequence should keep decoding.
func (r *Runner) stepSeq(st *seqState, row []float32) int {
	if st.cfg.Ctx != nil && st.cfg.Ctx.Err() != nil {
		st.finished, st.reason = true, "cancel"
		return 0
	}
	if st.cfg.Logits.Enabled() {
		st.cfg.Logits.Apply(row, st.hist)
	}
	tok := st.cfg.Sampler.Sample(row, st.rng)

	if r.eos[tok] {
		st.finished, st.reason = true, "stop"
		return 0
	}
	st.generated = append(st.generated, tok)
	st.hist = append(st.hist, tok)

	full := r.Tok.Decode(st.generated)
	if stop, cut := hitStopString(full, st.cfg.Stop); stop {
		if st.cfg.OnToken != nil && len(cut) > len(st.prevText) {
			st.cfg.OnToken(cut[len(st.prevText):])
		}
		st.finished, st.reason, st.text = true, "stop", cut
		return tok
	}
	if st.cfg.OnToken != nil && len(full) > len(st.prevText) {
		st.cfg.OnToken(full[len(st.prevText):])
	}
	st.prevText = full

	if len(st.generated) >= st.maxTokens {
		st.finished, st.reason = true, "length"
		return tok
	}
	return tok
}

// deliver sends the final result for a sequence to its waiting caller.
func (st *seqState) deliver(tok *tokenizer.Tokenizer) {
	if st.done == nil {
		return
	}
	if st.reason == "" {
		st.reason = "length"
	}
	text := st.text
	if text == "" {
		text = tok.Decode(st.generated)
	}
	st.done.done <- jobResult{res: GenResult{
		Text:             text,
		Tokens:           st.generated,
		PromptTokens:     len(st.done.prompt),
		CompletionTokens: len(st.generated),
		FinishReason:     st.reason,
	}}
}
