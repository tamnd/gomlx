// SPDX-License-Identifier: Apache-2.0

package routes

import (
	"net/http"
	"time"

	"github.com/tamnd/gomlx/api"
	"github.com/tamnd/gomlx/models"
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

// RetrieveModel handles GET /v1/models/{model_id}. It returns the model when the
// id names the served model, since a server loads one model and answers for it
// under any name that resolves to it: the configured name, the resolved path, or
// an alias pointing at the same path. An unknown id is a 404, matching the
// OpenAI contract.
func (d *Deps) RetrieveModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("model_id")
	if id == "" || !d.servesModel(id) {
		writeError(w, http.StatusNotFound, "model_not_found", "model "+id+" not found")
		return
	}
	writeJSON(w, http.StatusOK, api.ModelInfo{
		ID:      id,
		Object:  "model",
		Created: time.Now().Unix(),
		OwnedBy: "gomlx",
	})
}

// servesModel reports whether id refers to the model this server loaded. The
// configured name and the resolved path both match directly; any known alias
// matches when it resolves to the same path, so a client can ask by alias even
// when the server was started with the path, and the other way round.
func (d *Deps) servesModel(id string) bool {
	if id == d.Model {
		return true
	}
	target := models.ResolveModel(d.Model)
	if id == target {
		return true
	}
	return models.IsAlias(id) && models.ResolveModel(id) == target
}
