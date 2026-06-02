// SPDX-License-Identifier: Apache-2.0

package routes

import "net/http"

// Health handles GET /health and GET /v1/health.
func (d *Deps) Health(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"model":  d.Model,
	})
}
