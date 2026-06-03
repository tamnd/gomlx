// SPDX-License-Identifier: Apache-2.0

package agents

import "testing"

func TestVersionMatches(t *testing.T) {
	cases := []struct {
		version, spec string
		want          bool
	}{
		{"0.9.1", ">=0.9", true},
		{"0.8.9", ">=0.9", false},
		{"0.8.5", ">=0.8,<0.9", true},
		{"0.9.0", ">=0.8,<0.9", false},
		{"0.7.0", ">=0.8,<0.9", false},
		{"1.2.3", "1.2.3", true},
		{"1.2.3", "1.2.4", false},
		{"2.0", ">1.0", true},
		{"1.0", ">1.0", false},
		{"1.0", ">=1.0", true},
		{"1.0", "<=1.0", true},
		{"1.1", "<=1.0", false},
		{"1.2", "==1.2", true},
		{"1.2", "=1.2", true},
		{"1.2.0", "1.2", true}, // missing component compares as zero
		{"not-a-version", ">=1.0", false},
		{"1.0", "garbage", false},
	}
	for _, c := range cases {
		if got := versionMatches(c.version, c.spec); got != c.want {
			t.Errorf("versionMatches(%q, %q) = %v, want %v", c.version, c.spec, got, c.want)
		}
	}
}

func TestParseVersionLeadingDigits(t *testing.T) {
	got, ok := parseVersion("0.9-beta")
	if !ok || len(got) != 2 || got[0] != 0 || got[1] != 9 {
		t.Fatalf("parseVersion(0.9-beta) = %v ok=%v", got, ok)
	}
	if _, ok := parseVersion("beta"); ok {
		t.Fatal("parseVersion of a non-numeric string should fail")
	}
}

func TestConfigForVersionPicksOverride(t *testing.T) {
	h, _ := Get("hermes")
	// 0.8.x has its own config block; the rendered body still substitutes.
	old := h.ConfigForVersion("0.8.5")
	base := h.ConfigForVersion("")
	if old.Template == base.Template {
		t.Fatal("0.8.x config should differ from the base config")
	}
	// 0.9+ has only notes, so it falls back to the base config.
	if h.ConfigForVersion("0.9.1").Template != base.Template {
		t.Fatal("0.9.x should fall back to the base config")
	}
}
