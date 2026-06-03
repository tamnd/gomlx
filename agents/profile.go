// SPDX-License-Identifier: Apache-2.0

// Package agents holds declarative profiles describing what each known agent
// framework needs from this inference server: how to point it at the API, which
// models it works well with, its streaming quirks, and its known issues. A
// profile is data, so supporting a new agent is a matter of adding one entry
// rather than writing code. Profiles are versioned: when an agent changes its
// config format in a new release, a version block overrides the base rather than
// breaking the old shape.
package agents

import "strings"

// ConfigSpec is how to configure an agent to reach this server. Type is one of
// "env", "yaml", "json", or "toml". For "env" the EnvVars are rendered; for the
// file types the Template is rendered to the file at Path.
type ConfigSpec struct {
	Type     string
	Path     string
	Template string
	EnvVars  map[string]string
}

// TagPair is an open/close marker an agent wraps around tool calls, which the
// streaming parser must recognize beyond the model's own format.
type TagPair struct {
	Open  string
	Close string
}

// StreamingSpec captures an agent's streaming behavior. MaxTools is the number
// of tools the agent injects per request, zero when unspecified.
type StreamingSpec struct {
	ExtraToolTags    []TagPair
	SuppressPatterns []string
	MaxTools         int
}

// TestingSpec is the integration-testing recipe for an agent.
type TestingSpec struct {
	Binary        string
	QueryCmd      string
	QueryTimeout  int
	InstallCmd    string
	SpecificTests string
}

// VersionSpec overrides parts of a profile for a range of agent versions. A nil
// section means that version keeps the base profile's value.
type VersionSpec struct {
	Range     string
	Notes     string
	Config    *ConfigSpec
	Streaming *StreamingSpec
	Testing   *TestingSpec
}

// Profile is everything needed to integrate one agent framework.
type Profile struct {
	Name        string
	DisplayName string
	Repo        string
	Stars       int

	// Config is the base configuration, used when no version block matches.
	Config            ConfigSpec
	RecommendedModels []string
	ParserOverride    string

	Streaming   StreamingSpec
	Testing     TestingSpec
	Versions    []VersionSpec
	KnownIssues []string

	NeedsFunctionCalling bool
	NeedsStreaming       bool
	NeedsVision          bool
	NeedsReasoning       bool
}

// ConfigForVersion returns the config spec, preferring the first version block
// whose range matches version and that overrides the config. An empty version,
// no version blocks, or no match falls back to the base config.
func (p Profile) ConfigForVersion(version string) ConfigSpec {
	if version != "" {
		for _, vs := range p.Versions {
			if vs.Config != nil && versionMatches(version, vs.Range) {
				return *vs.Config
			}
		}
	}
	return p.Config
}

// StreamingForVersion returns the streaming spec for version, falling back to
// the base spec the same way ConfigForVersion does.
func (p Profile) StreamingForVersion(version string) StreamingSpec {
	if version != "" {
		for _, vs := range p.Versions {
			if vs.Streaming != nil && versionMatches(version, vs.Range) {
				return *vs.Streaming
			}
		}
	}
	return p.Streaming
}

// TestingForVersion returns the testing spec for version, falling back to the
// base spec.
func (p Profile) TestingForVersion(version string) TestingSpec {
	if version != "" {
		for _, vs := range p.Versions {
			if vs.Testing != nil && versionMatches(version, vs.Range) {
				return *vs.Testing
			}
		}
	}
	return p.Testing
}

// RenderedConfig is the result of rendering a profile against a live server.
// For an env-typed config EnvVars is filled; for a file-typed config Content
// holds the rendered file body.
type RenderedConfig struct {
	Type    string
	EnvVars map[string]string
	Content string
}

// RenderConfig fills the placeholders in the profile's config for version with
// the running server's base URL and model. The recognized placeholders are
// {base_url}, {model_id}, and {base_url_no_v1}, the last being base_url with a
// trailing "/v1" removed for agents that add their own version segment.
func (p Profile) RenderConfig(baseURL, modelID, version string) RenderedConfig {
	cfg := p.ConfigForVersion(version)
	baseNoV1 := strings.TrimSuffix(strings.TrimRight(baseURL, "/"), "/v1")

	subst := func(s string) string {
		s = strings.ReplaceAll(s, "{base_url}", baseURL)
		s = strings.ReplaceAll(s, "{model_id}", modelID)
		s = strings.ReplaceAll(s, "{base_url_no_v1}", baseNoV1)
		return s
	}

	if cfg.Type == "env" {
		env := make(map[string]string, len(cfg.EnvVars))
		for k, v := range cfg.EnvVars {
			env[k] = subst(v)
		}
		return RenderedConfig{Type: cfg.Type, EnvVars: env}
	}
	return RenderedConfig{Type: cfg.Type, Content: subst(cfg.Template)}
}
