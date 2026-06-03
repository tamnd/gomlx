// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"fmt"
	"sort"
	"sync"
)

// The registry is where the tools discovered on every connected server are
// pooled and handed to the model as one flat, namespaced list. It is also where
// the security gate is enforced in practice: a high-risk tool is left out of the
// list the model sees, and a call that slips through is refused at resolve time.
// A server can opt out of the gate when it is explicitly trusted, which is how the
// per-server SkipSecurityValidation flag from the config takes effect here.

// Registry holds the tools offered by each connected server and applies the
// security policy from the config. It is safe for concurrent use.
type Registry struct {
	allowedHighRisk []string

	mu       sync.RWMutex
	byServer map[string][]Tool
	skip     map[string]bool // servers that bypass the security gate
}

// NewRegistry returns a registry governed by cfg. The allowlist and the
// per-server skip flags are read from cfg up front.
func NewRegistry(cfg Config) *Registry {
	skip := make(map[string]bool, len(cfg.Servers))
	for name, s := range cfg.Servers {
		if s.SkipSecurityValidation {
			skip[name] = true
		}
	}
	return &Registry{
		allowedHighRisk: cfg.AllowedHighRiskTools,
		byServer:        make(map[string][]Tool),
		skip:            skip,
	}
}

// SetTools records the tools a server offers, replacing any it offered before.
func (r *Registry) SetTools(serverName string, tools []Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	clone := make([]Tool, len(tools))
	copy(clone, tools)
	for i := range clone {
		clone[i].ServerName = serverName
	}
	r.byServer[serverName] = clone
}

// RemoveServer drops a server's tools, as when it disconnects.
func (r *Registry) RemoveServer(serverName string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.byServer, serverName)
}

// Tools returns every registered tool, sorted by namespaced name for a stable
// order regardless of server connection order.
func (r *Registry) Tools() []Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var all []Tool
	for _, tools := range r.byServer {
		all = append(all, tools...)
	}
	sort.Slice(all, func(i, j int) bool {
		return all[i].FullName() < all[j].FullName()
	})
	return all
}

// OpenAITools returns the registered tools in OpenAI format, leaving out any tool
// the security gate refuses. Hiding a blocked tool rather than offering it and
// failing later keeps the model from planning a call it can never make.
func (r *Registry) OpenAITools() []map[string]any {
	tools := r.Tools()
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if r.toolAllowed(t) {
			out = append(out, t.ToOpenAI())
		}
	}
	return out
}

// Resolve looks up a tool by its namespaced name and runs the call through the
// security gate. It returns the tool when the call is permitted, or an error when
// the tool is unknown or the gate refuses it. A server marked to skip validation
// bypasses both checks.
func (r *Registry) Resolve(fullName string, args map[string]any) (Tool, error) {
	r.mu.RLock()
	tool, found := r.lookup(fullName)
	skip := found && r.skip[tool.ServerName]
	r.mu.RUnlock()

	if !found {
		return Tool{}, fmt.Errorf("mcp: unknown tool %q", fullName)
	}
	if skip {
		return tool, nil
	}
	if err := CheckTool(tool.Name, tool.FullName(), r.allowedHighRisk); err != nil {
		return Tool{}, err
	}
	if err := CheckArgs(tool.FullName(), args); err != nil {
		return Tool{}, err
	}
	return tool, nil
}

// toolAllowed reports whether the tool clears the security gate for listing. It
// considers only the tool name, since arguments are not known until a call is
// made; a trusted server's tools are always listed.
func (r *Registry) toolAllowed(t Tool) bool {
	r.mu.RLock()
	skip := r.skip[t.ServerName]
	r.mu.RUnlock()
	if skip {
		return true
	}
	return CheckTool(t.Name, t.FullName(), r.allowedHighRisk) == nil
}

// lookup finds a tool by namespaced name. The caller holds the lock.
func (r *Registry) lookup(fullName string) (Tool, bool) {
	for _, tools := range r.byServer {
		for _, t := range tools {
			if t.FullName() == fullName {
				return t, true
			}
		}
	}
	return Tool{}, false
}
