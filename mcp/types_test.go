// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"slices"
	"testing"
)

func TestParseConfigDefaults(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"servers": {
			"files": {"command": "mcp-files", "args": ["--root", "."]}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.MaxToolCalls != DefaultMaxToolCalls {
		t.Fatalf("max_tool_calls=%d want default %d", cfg.MaxToolCalls, DefaultMaxToolCalls)
	}
	if cfg.DefaultTimeoutSeconds != DefaultTimeoutSeconds {
		t.Fatalf("default_timeout=%v want %v", cfg.DefaultTimeoutSeconds, DefaultTimeoutSeconds)
	}

	s, ok := cfg.Servers["files"]
	if !ok {
		t.Fatal("server files missing")
	}
	if s.Name != "files" {
		t.Fatalf("name not filled from key: %q", s.Name)
	}
	if s.Transport != TransportStdio {
		t.Fatalf("transport=%q want default stdio", s.Transport)
	}
	if !s.Enabled {
		t.Fatal("a server with no enabled field should default to enabled")
	}
	if s.TimeoutSeconds != DefaultTimeoutSeconds {
		t.Fatalf("server timeout=%v should inherit the default", s.TimeoutSeconds)
	}
	if !slices.Equal(s.Args, []string{"--root", "."}) {
		t.Fatalf("args=%v", s.Args)
	}
}

func TestParseConfigEnabledFalseRespected(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"servers": {"off": {"command": "x", "enabled": false}}
	}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.Servers["off"].Enabled {
		t.Fatal("enabled:false should be honored, not overwritten by the default")
	}
}

func TestParseConfigOverrides(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{
		"max_tool_calls": 3,
		"default_timeout": 5,
		"allowed_high_risk_tools": ["files.exec"],
		"servers": {
			"remote": {"transport": "sse", "url": "https://h/sse", "timeout": 12}
		}
	}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.MaxToolCalls != 3 || cfg.DefaultTimeoutSeconds != 5 {
		t.Fatalf("overrides not applied: %+v", cfg)
	}
	if !slices.Equal(cfg.AllowedHighRiskTools, []string{"files.exec"}) {
		t.Fatalf("allowlist=%v", cfg.AllowedHighRiskTools)
	}
	s := cfg.Servers["remote"]
	if s.Transport != TransportSSE || s.URL != "https://h/sse" {
		t.Fatalf("sse server wrong: %+v", s)
	}
	if s.TimeoutSeconds != 12 {
		t.Fatalf("explicit timeout=%v want 12", s.TimeoutSeconds)
	}
}

func TestParseConfigRejectsBadTransport(t *testing.T) {
	cases := map[string]string{
		"stdio missing command": `{"servers": {"a": {"transport": "stdio"}}}`,
		"sse missing url":       `{"servers": {"a": {"transport": "sse"}}}`,
		"unknown transport":     `{"servers": {"a": {"transport": "carrier", "command": "x"}}}`,
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseConfig([]byte(in)); err == nil {
				t.Fatal("expected a validation error")
			}
		})
	}
}

func TestParseConfigRejectsBadJSON(t *testing.T) {
	if _, err := ParseConfig([]byte(`{not json`)); err == nil {
		t.Fatal("expected a parse error")
	}
}
