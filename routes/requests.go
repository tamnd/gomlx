// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"context"
	"net/http"
	"sync"
)

// cancelRegistry tracks the cancel funcs of in-flight streaming requests, keyed
// by the response id the client sees in the stream. It lets POST
// /v1/requests/{id}/cancel stop a stream that is still running. The zero value
// is ready to use; the map is created on first registration so a Deps built
// without any wiring still works.
//
// Only streaming requests register: a non-streaming call has already returned by
// the time a cancel could arrive, so there is nothing to stop.
type cancelRegistry struct {
	mu      sync.Mutex
	cancels map[string]context.CancelFunc
}

// add records the cancel func for id.
func (c *cancelRegistry) add(id string, fn context.CancelFunc) {
	c.mu.Lock()
	if c.cancels == nil {
		c.cancels = make(map[string]context.CancelFunc)
	}
	c.cancels[id] = fn
	c.mu.Unlock()
}

// remove forgets id without cancelling it, called when a stream finishes on its
// own.
func (c *cancelRegistry) remove(id string) {
	c.mu.Lock()
	delete(c.cancels, id)
	c.mu.Unlock()
}

// count reports how many requests are in flight.
func (c *cancelRegistry) count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.cancels)
}

// cancel stops the request for id and forgets it, returning false when no such
// request is in flight.
func (c *cancelRegistry) cancel(id string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	fn, ok := c.cancels[id]
	if !ok {
		return false
	}
	fn()
	delete(c.cancels, id)
	return true
}

// trackStream derives a cancellable context from ctx and registers it under id,
// so a concurrent cancel request can stop the stream. The returned stop func
// deregisters id and releases the context; defer it where the stream ends.
func (d *Deps) trackStream(ctx context.Context, id string) (context.Context, func()) {
	ctx, cancel := context.WithCancel(ctx)
	d.Cancels.add(id, cancel)
	return ctx, func() {
		d.Cancels.remove(id)
		cancel()
	}
}

// CancelRequest handles POST /v1/requests/{request_id}/cancel and its DELETE
// alias. The request_id is the id returned in the response body or the first
// streaming chunk. A request that is not in flight, has already finished, or was
// never streaming is a 404.
func (d *Deps) CancelRequest(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("request_id")
	if id == "" || !d.Cancels.cancel(id) {
		writeError(w, http.StatusNotFound, "not_found",
			"request "+id+" not found or already finished")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object":    "request.cancel",
		"id":        id,
		"cancelled": true,
		"model":     d.Model,
	})
}
