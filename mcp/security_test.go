// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"errors"
	"testing"
)

func TestCheckToolBlocksHighRisk(t *testing.T) {
	risky := []string{"execute", "run_command", "shell_open", "do_eval", "exec", "system_call", "subprocess_run", "EXECUTE"}
	for _, name := range risky {
		if err := CheckTool(name, "srv."+name, nil); err == nil {
			t.Fatalf("tool %q should be refused", name)
		}
	}
}

func TestCheckToolAllowsSafe(t *testing.T) {
	for _, name := range []string{"read_file", "list_dir", "search", "fetch_url"} {
		if err := CheckTool(name, "srv."+name, nil); err != nil {
			t.Fatalf("tool %q should be allowed: %v", name, err)
		}
	}
}

func TestCheckToolAllowlist(t *testing.T) {
	// Bare name allowlisted.
	if err := CheckTool("execute", "srv.execute", []string{"execute"}); err != nil {
		t.Fatalf("allowlisted bare name should pass: %v", err)
	}
	// Namespaced name allowlisted.
	if err := CheckTool("execute", "srv.execute", []string{"srv.execute"}); err != nil {
		t.Fatalf("allowlisted full name should pass: %v", err)
	}
	// A different entry does not open the gate.
	if err := CheckTool("execute", "srv.execute", []string{"other.execute"}); err == nil {
		t.Fatal("an unrelated allowlist entry should not clear the tool")
	}
}

func TestCheckToolErrorType(t *testing.T) {
	err := CheckTool("shell", "srv.shell", nil)
	var se *SecurityError
	if !errors.As(err, &se) {
		t.Fatalf("want *SecurityError, got %T", err)
	}
	if se.Tool != "srv.shell" {
		t.Fatalf("error should carry the full name, got %q", se.Tool)
	}
}

func TestCheckArgsBlocksDangerousPaths(t *testing.T) {
	cases := []map[string]any{
		{"path": "../../etc/shadow"},
		{"file": "/etc/passwd"},
		{"target": "/proc/self/mem"},
		{"dir": "/sys/kernel"},
		{"home": "~root/.ssh"},
		{"p": "/root/secrets"},
	}
	for _, args := range cases {
		if err := CheckArgs("read_file", args); err == nil {
			t.Fatalf("args %v should be refused", args)
		}
	}
}

func TestCheckArgsWalksNestedValues(t *testing.T) {
	args := map[string]any{
		"options": map[string]any{
			"paths": []any{"ok/relative", "../escape"},
		},
	}
	if err := CheckArgs("tool", args); err == nil {
		t.Fatal("a dangerous path nested in a list inside a map should be caught")
	}
}

func TestCheckArgsAllowsCleanValues(t *testing.T) {
	args := map[string]any{
		"path":  "data/input.txt",
		"count": 5,
		"opts":  map[string]any{"recursive": true, "globs": []any{"*.go", "src/**"}},
	}
	if err := CheckArgs("tool", args); err != nil {
		t.Fatalf("clean args should pass: %v", err)
	}
}
