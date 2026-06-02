// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"net/http"
	"time"

	"github.com/tamnd/gomlx/api"
)

// Models handles GET /v1/models, listing the served model.
func (d *Deps) Models(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, api.ModelsResponse{
		Object: "list",
		Data: []api.ModelInfo{{
			ID:      d.Model,
			Object:  "model",
			Created: time.Now().Unix(),
			OwnedBy: "gomlx",
		}},
	})
}
