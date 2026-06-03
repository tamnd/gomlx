// SPDX-License-Identifier: Apache-2.0

// Package mcp speaks the Model Context Protocol, which lets the server pull tools
// and context from external MCP servers and offer them to the model. This file
// holds the configuration the rest of the package is driven by: which servers to
// connect to, over which transport, and the safety limits on tool use. The
// connection clients and the tool executor build on these types; the security
// gate that decides whether a given tool call is allowed lives alongside in
// security.go.
package mcp

import (
	"encoding/json"
	"fmt"
)

// Transport is how the client talks to an MCP server.
type Transport string

const (
	// TransportStdio launches the server as a subprocess and exchanges
	// JSON-RPC over its standard input and output.
	TransportStdio Transport = "stdio"
	// TransportSSE connects to a server over HTTP using server-sent events.
	TransportSSE Transport = "sse"
)

// ServerState is a server connection's lifecycle state.
type ServerState string

const (
	StateDisconnected ServerState = "disconnected"
	StateConnecting   ServerState = "connecting"
	StateConnected    ServerState = "connected"
	StateError        ServerState = "error"
)

// Default limits applied when a config omits them.
const (
	DefaultMaxToolCalls   = 10
	DefaultTimeoutSeconds = 30.0
)

// ServerConfig describes one MCP server to connect to.
type ServerConfig struct {
	Name      string    `json:"-"`
	Transport Transport `json:"transport"`

	// Stdio transport.
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`

	// SSE transport.
	URL string `json:"url,omitempty"`

	Enabled        bool    `json:"enabled"`
	TimeoutSeconds float64 `json:"timeout,omitempty"`

	// SkipSecurityValidation disables the safety checks for this server. It is
	// for local development only and should never be set against a server you
	// do not control.
	SkipSecurityValidation bool `json:"skip_security_validation,omitempty"`
}

// Validate checks that the transport has the fields it needs: stdio needs a
// command, SSE needs a URL.
func (c ServerConfig) Validate() error {
	switch c.Transport {
	case TransportStdio:
		if c.Command == "" {
			return fmt.Errorf("mcp server %q: stdio transport requires a command", c.Name)
		}
	case TransportSSE:
		if c.URL == "" {
			return fmt.Errorf("mcp server %q: sse transport requires a url", c.Name)
		}
	default:
		return fmt.Errorf("mcp server %q: unknown transport %q", c.Name, c.Transport)
	}
	return nil
}

// Config is the root MCP configuration.
type Config struct {
	Servers               map[string]ServerConfig
	MaxToolCalls          int
	DefaultTimeoutSeconds float64

	// AllowedHighRiskTools opts specific tools past the high-risk block. A tool
	// whose name matches a high-risk pattern is refused unless its bare or
	// namespaced name appears here.
	AllowedHighRiskTools []string
}

// wireConfig mirrors the JSON shape. Enabled is a pointer so an omitted field
// defaults to true rather than the zero value false.
type wireConfig struct {
	Servers map[string]struct {
		Transport              Transport         `json:"transport"`
		Command                string            `json:"command"`
		Args                   []string          `json:"args"`
		Env                    map[string]string `json:"env"`
		URL                    string            `json:"url"`
		Enabled                *bool             `json:"enabled"`
		TimeoutSeconds         float64           `json:"timeout"`
		SkipSecurityValidation bool              `json:"skip_security_validation"`
	} `json:"servers"`
	MaxToolCalls          int      `json:"max_tool_calls"`
	DefaultTimeoutSeconds float64  `json:"default_timeout"`
	AllowedHighRiskTools  []string `json:"allowed_high_risk_tools"`
}

// ParseConfig reads an MCP config from JSON, filling defaults and validating each
// server. A server with no transport defaults to stdio, and a server is enabled
// unless it sets "enabled": false.
func ParseConfig(data []byte) (Config, error) {
	var w wireConfig
	if err := json.Unmarshal(data, &w); err != nil {
		return Config{}, fmt.Errorf("mcp: parse config: %w", err)
	}

	cfg := Config{
		Servers:               make(map[string]ServerConfig, len(w.Servers)),
		MaxToolCalls:          w.MaxToolCalls,
		DefaultTimeoutSeconds: w.DefaultTimeoutSeconds,
		AllowedHighRiskTools:  w.AllowedHighRiskTools,
	}
	if cfg.MaxToolCalls <= 0 {
		cfg.MaxToolCalls = DefaultMaxToolCalls
	}
	if cfg.DefaultTimeoutSeconds <= 0 {
		cfg.DefaultTimeoutSeconds = DefaultTimeoutSeconds
	}

	for name, s := range w.Servers {
		transport := s.Transport
		if transport == "" {
			transport = TransportStdio
		}
		enabled := true
		if s.Enabled != nil {
			enabled = *s.Enabled
		}
		timeout := s.TimeoutSeconds
		if timeout <= 0 {
			timeout = cfg.DefaultTimeoutSeconds
		}
		sc := ServerConfig{
			Name:                   name,
			Transport:              transport,
			Command:                s.Command,
			Args:                   s.Args,
			Env:                    s.Env,
			URL:                    s.URL,
			Enabled:                enabled,
			TimeoutSeconds:         timeout,
			SkipSecurityValidation: s.SkipSecurityValidation,
		}
		if err := sc.Validate(); err != nil {
			return Config{}, err
		}
		cfg.Servers[name] = sc
	}
	return cfg, nil
}
