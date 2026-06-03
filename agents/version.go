// SPDX-License-Identifier: Apache-2.0

package agents

import (
	"strconv"
	"strings"
)

// versionMatches reports whether version satisfies the comma-separated range
// spec. Each constraint is one of ">=x", ">x", "<=x", "<x", "==x"/"=x", or a
// bare version meaning an exact match. All constraints must hold. A version or
// constraint that does not parse causes no match.
func versionMatches(version, spec string) bool {
	parts, ok := parseVersion(version)
	if !ok {
		return false
	}
	for c := range strings.SplitSeq(spec, ",") {
		c = strings.TrimSpace(c)
		switch {
		case strings.HasPrefix(c, ">="):
			if t, ok := parseVersion(c[2:]); ok && compareVersion(parts, t) < 0 {
				return false
			}
		case strings.HasPrefix(c, "<="):
			if t, ok := parseVersion(c[2:]); ok && compareVersion(parts, t) > 0 {
				return false
			}
		case strings.HasPrefix(c, ">"):
			if t, ok := parseVersion(c[1:]); ok && compareVersion(parts, t) <= 0 {
				return false
			}
		case strings.HasPrefix(c, "<"):
			if t, ok := parseVersion(c[1:]); ok && compareVersion(parts, t) >= 0 {
				return false
			}
		case strings.HasPrefix(c, "=="), strings.HasPrefix(c, "="):
			if t, ok := parseVersion(strings.TrimLeft(c, "=")); ok && compareVersion(parts, t) != 0 {
				return false
			}
		default:
			t, ok := parseVersion(c)
			if !ok || compareVersion(parts, t) != 0 {
				return false
			}
		}
	}
	return true
}

// parseVersion reads the leading dotted-integer run of v into its components,
// so "1.2.3" becomes {1, 2, 3} and "0.9-beta" becomes {0, 9}. It returns false
// when v has no leading integer.
func parseVersion(v string) ([]int, bool) {
	v = strings.TrimSpace(v)
	end := 0
	for end < len(v) && (v[end] == '.' || (v[end] >= '0' && v[end] <= '9')) {
		end++
	}
	if end == 0 {
		return nil, false
	}
	var out []int
	for seg := range strings.SplitSeq(v[:end], ".") {
		if seg == "" {
			continue
		}
		n, err := strconv.Atoi(seg)
		if err != nil {
			return nil, false
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, false
	}
	return out, true
}

// compareVersion orders two parsed versions component by component, treating a
// missing trailing component as zero so "1.2" equals "1.2.0". It returns -1, 0,
// or 1.
func compareVersion(a, b []int) int {
	n := max(len(a), len(b))
	for i := range n {
		var ai, bi int
		if i < len(a) {
			ai = a[i]
		}
		if i < len(b) {
			bi = b[i]
		}
		if ai != bi {
			if ai < bi {
				return -1
			}
			return 1
		}
	}
	return 0
}
