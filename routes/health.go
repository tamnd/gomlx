// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"net/http"

	"github.com/tamnd/gomlx/mcp"
)

// Health handles GET /health and GET /v1/health.
func (d *Deps) Health(w http.ResponseWriter, r *http.Request) {
	body := map[string]any{
		"status": "ok",
		"model":  d.Model,
	}
	if d.MCP != nil {
		statuses := d.MCP.Statuses()
		connected := 0
		for _, s := range statuses {
			if s.State == mcp.StateConnected {
				connected++
			}
		}
		body["mcp"] = map[string]any{
			"servers":         len(statuses),
			"connected":       connected,
			"tools_available": len(d.MCP.Registry().Tools()),
		}
	}
	writeJSON(w, http.StatusOK, body)
}
