// SPDX-License-Identifier: Apache-2.0

// Package config holds server and runtime configuration shared across packages.
package config

// ServerConfig is the top-level server configuration assembled from CLI flags,
// environment, and defaults. It grows as the serving layer lands (stage 2).
type ServerConfig struct {
	Host  string
	Port  int
	Model string // alias or hf_path

	// Parser overrides; empty means auto-detect via models.DetectModelConfig.
	ToolCallParser  string
	ReasoningParser string

	// Generation defaults.
	MaxTokens int

	// Serving knobs.
	MaxConcurrent int    // max concurrent requests admitted to the engine
	APIKey        string // optional bearer key; empty disables auth

	// MCPConfig is the path to a JSON file describing the MCP servers to
	// connect to. Empty disables the MCP subsystem.
	MCPConfig string

	// Mock runs the serving layer with the mock decode backend (no GPU). Used
	// before the compute backend (stage 4) lands.
	Mock bool
}

// Default returns a ServerConfig with reference-aligned defaults.
func Default() ServerConfig {
	return ServerConfig{
		Host:          "127.0.0.1",
		Port:          8000,
		MaxTokens:     256,
		MaxConcurrent: 256,
	}
}
