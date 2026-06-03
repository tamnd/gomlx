// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"fmt"
	"strings"
)

// An MCP server can offer the model any tool it likes, and the model can be
// talked into calling one. This file is the gate that stands between a tool call
// the model wants to make and actually making it. Two kinds of danger get
// stopped: tools that can run arbitrary code or commands, and tool arguments that
// reach for sensitive paths. Both checks are deny-by-default for the matching
// cases, with an explicit allowlist for the rare legitimate use.

// highRiskToolPatterns are substrings that mark a tool as able to run arbitrary
// code or shell commands. A tool whose name contains any of these is refused
// unless it is explicitly allowlisted.
var highRiskToolPatterns = []string{
	"execute",
	"run_command",
	"shell",
	"eval",
	"exec",
	"system",
	"subprocess",
}

// dangerousArgPatterns are substrings in a tool argument that point at sensitive
// filesystem locations or attempt to climb out of a working directory.
var dangerousArgPatterns = []string{
	"../",
	"/etc/",
	"/proc/",
	"/sys/",
	"~root",
	"/root/",
}

// SecurityError reports that a tool call was refused by the security gate.
type SecurityError struct {
	Tool   string
	Reason string
}

func (e *SecurityError) Error() string {
	return fmt.Sprintf("mcp: tool %q refused: %s", e.Tool, e.Reason)
}

// CheckTool refuses a tool whose name marks it as high risk unless it has been
// allowlisted. toolName is the bare tool name and fullName is the namespaced
// "server.tool" form; a match on either against the allowlist clears the tool.
// allowed is the AllowedHighRiskTools list from the config.
func CheckTool(toolName, fullName string, allowed []string) error {
	lower := strings.ToLower(toolName)
	for _, pattern := range highRiskToolPatterns {
		if strings.Contains(lower, pattern) {
			if isAllowed(toolName, fullName, allowed) {
				return nil
			}
			return &SecurityError{
				Tool:   fullName,
				Reason: fmt.Sprintf("matches high-risk pattern %q and is not in the allowlist", pattern),
			}
		}
	}
	return nil
}

// CheckArgs refuses a tool call whose arguments reach for a sensitive path. It
// walks the argument tree, so a dangerous string nested inside a list or object
// is caught too. toolName is used only for the error message.
func CheckArgs(toolName string, args map[string]any) error {
	return checkValue(toolName, args)
}

func checkValue(toolName string, v any) error {
	switch val := v.(type) {
	case string:
		if pattern, ok := matchDangerous(val); ok {
			return &SecurityError{
				Tool:   toolName,
				Reason: fmt.Sprintf("argument references sensitive path pattern %q", pattern),
			}
		}
	case map[string]any:
		for _, item := range val {
			if err := checkValue(toolName, item); err != nil {
				return err
			}
		}
	case []any:
		for _, item := range val {
			if err := checkValue(toolName, item); err != nil {
				return err
			}
		}
	}
	return nil
}

// matchDangerous reports the first dangerous pattern contained in s.
func matchDangerous(s string) (string, bool) {
	lower := strings.ToLower(s)
	for _, pattern := range dangerousArgPatterns {
		if strings.Contains(lower, pattern) {
			return pattern, true
		}
	}
	return "", false
}

// isAllowed reports whether the bare or namespaced tool name is in the allowlist.
func isAllowed(toolName, fullName string, allowed []string) bool {
	for _, a := range allowed {
		if a == toolName || a == fullName {
			return true
		}
	}
	return false
}
