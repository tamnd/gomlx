// SPDX-License-Identifier: Apache-2.0

package routes

import "net/http"

// statusResponse is the body of GET /v1/status. It reports what the serving
// layer knows for certain: which model is loaded and how many streaming
// requests are in flight right now. Runtime metrics that only the compute
// backend can measure, such as tokens per second and device memory, are added
// when that backend lands rather than reported as zeros here.
type statusResponse struct {
	Status     string `json:"status"`
	Model      string `json:"model"`
	NumRunning int    `json:"num_running"`
}

// Status handles GET /v1/status, a real-time snapshot of the server. The status
// is "generating" while any stream is in flight and "idle" otherwise.
func (d *Deps) Status(w http.ResponseWriter, r *http.Request) {
	running := d.Cancels.count()
	state := "idle"
	if running > 0 {
		state = "generating"
	}
	writeJSON(w, http.StatusOK, statusResponse{
		Status:     state,
		Model:      d.Model,
		NumRunning: running,
	})
}
