// SPDX-License-Identifier: Apache-2.0

package agents

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
	"sync"
)

// profilesJSON holds the built-in agent profiles, keyed by name. It mirrors the
// section layout the profiles are authored in (config, models, streaming,
// testing, capabilities, versions) and is flattened into Profile on load.
//
//go:embed profiles.json
var profilesJSON []byte

var (
	loadOnce sync.Once
	loadErr  error
	profiles map[string]Profile
)

// rawProfile is the on-disk shape of one profile, before flattening into the
// Profile the rest of the package uses.
type rawProfile struct {
	Name        string    `json:"name"`
	DisplayName string    `json:"display_name"`
	Repo        string    `json:"repo"`
	Stars       int       `json:"stars"`
	Config      rawConfig `json:"config"`
	Models      struct {
		Recommended    []string `json:"recommended"`
		ParserOverride string   `json:"parser_override"`
	} `json:"models"`
	Streaming    rawStreaming `json:"streaming"`
	Testing      rawTesting   `json:"testing"`
	Versions     []rawVersion `json:"versions"`
	KnownIssues  []string     `json:"known_issues"`
	Capabilities struct {
		FunctionCalling *bool `json:"function_calling"`
		Streaming       *bool `json:"streaming"`
		Vision          *bool `json:"vision"`
		Reasoning       *bool `json:"reasoning"`
	} `json:"capabilities"`
}

type rawConfig struct {
	Type     string            `json:"type"`
	Path     string            `json:"path"`
	Template string            `json:"template"`
	EnvVars  map[string]string `json:"env_vars"`
}

type rawStreaming struct {
	ExtraToolTags    [][]string `json:"extra_tool_tags"`
	SuppressPatterns []string   `json:"suppress_patterns"`
	MaxTools         int        `json:"max_tools"`
}

type rawTesting struct {
	Binary        string `json:"binary"`
	QueryCmd      string `json:"query_cmd"`
	QueryTimeout  *int   `json:"query_timeout"`
	InstallCmd    string `json:"install_cmd"`
	SpecificTests string `json:"specific_tests"`
}

type rawVersion struct {
	Range     string        `json:"range"`
	Notes     string        `json:"notes"`
	Config    *rawConfig    `json:"config"`
	Streaming *rawStreaming `json:"streaming"`
	Testing   *rawTesting   `json:"testing"`
}

// defaultQueryTimeout is the testing timeout applied when a profile omits one.
const defaultQueryTimeout = 120

func toConfig(r rawConfig) ConfigSpec {
	typ := r.Type
	if typ == "" {
		typ = "env"
	}
	return ConfigSpec{Type: typ, Path: r.Path, Template: r.Template, EnvVars: r.EnvVars}
}

func toStreaming(r rawStreaming) StreamingSpec {
	tags := make([]TagPair, 0, len(r.ExtraToolTags))
	for _, t := range r.ExtraToolTags {
		switch len(t) {
		case 1:
			tags = append(tags, TagPair{Open: t[0]})
		default:
			if len(t) >= 2 {
				tags = append(tags, TagPair{Open: t[0], Close: t[1]})
			}
		}
	}
	return StreamingSpec{ExtraToolTags: tags, SuppressPatterns: r.SuppressPatterns, MaxTools: r.MaxTools}
}

func toTesting(r rawTesting) TestingSpec {
	timeout := defaultQueryTimeout
	if r.QueryTimeout != nil {
		timeout = *r.QueryTimeout
	}
	return TestingSpec{
		Binary:        r.Binary,
		QueryCmd:      r.QueryCmd,
		QueryTimeout:  timeout,
		InstallCmd:    r.InstallCmd,
		SpecificTests: r.SpecificTests,
	}
}

func boolOr(p *bool, def bool) bool {
	if p != nil {
		return *p
	}
	return def
}

// toProfile flattens a raw profile, mirroring how the authored sections map onto
// the flat Profile: models and capabilities are folded in, and version blocks
// keep only the sections they actually override.
func toProfile(r rawProfile) Profile {
	display := r.DisplayName
	if display == "" {
		display = r.Name
	}
	versions := make([]VersionSpec, 0, len(r.Versions))
	for _, v := range r.Versions {
		vs := VersionSpec{Range: v.Range, Notes: v.Notes}
		if v.Config != nil {
			c := toConfig(*v.Config)
			vs.Config = &c
		}
		if v.Streaming != nil {
			s := toStreaming(*v.Streaming)
			vs.Streaming = &s
		}
		if v.Testing != nil {
			t := toTesting(*v.Testing)
			vs.Testing = &t
		}
		versions = append(versions, vs)
	}
	return Profile{
		Name:                 r.Name,
		DisplayName:          display,
		Repo:                 r.Repo,
		Stars:                r.Stars,
		Config:               toConfig(r.Config),
		RecommendedModels:    r.Models.Recommended,
		ParserOverride:       r.Models.ParserOverride,
		Streaming:            toStreaming(r.Streaming),
		Testing:              toTesting(r.Testing),
		Versions:             versions,
		KnownIssues:          r.KnownIssues,
		NeedsFunctionCalling: boolOr(r.Capabilities.FunctionCalling, true),
		NeedsStreaming:       boolOr(r.Capabilities.Streaming, true),
		NeedsVision:          boolOr(r.Capabilities.Vision, false),
		NeedsReasoning:       boolOr(r.Capabilities.Reasoning, false),
	}
}

func load() {
	var raw map[string]rawProfile
	if err := json.Unmarshal(profilesJSON, &raw); err != nil {
		loadErr = fmt.Errorf("agents: parse embedded profiles: %w", err)
		return
	}
	profiles = make(map[string]Profile, len(raw))
	for name, r := range raw {
		if r.Name == "" {
			r.Name = name
		}
		profiles[r.Name] = toProfile(r)
	}
}

func ensureLoaded() {
	loadOnce.Do(load)
}

// Get returns the profile registered under name. The bool is false when no such
// profile exists or the embedded data failed to parse.
func Get(name string) (Profile, bool) {
	ensureLoaded()
	p, ok := profiles[name]
	return p, ok
}

// GetOrGeneric returns the named profile, falling back to the "generic" profile
// when the name is unknown. The bool is false only when neither exists.
func GetOrGeneric(name string) (Profile, bool) {
	ensureLoaded()
	if p, ok := profiles[name]; ok {
		return p, true
	}
	p, ok := profiles["generic"]
	return p, ok
}

// List returns every profile, ordered by star count descending and then by name
// so the order is stable.
func List() []Profile {
	ensureLoaded()
	out := make([]Profile, 0, len(profiles))
	for _, p := range profiles {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Stars != out[j].Stars {
			return out[i].Stars > out[j].Stars
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// Names returns the names of every registered profile, sorted.
func Names() []string {
	ensureLoaded()
	out := make([]string, 0, len(profiles))
	for name := range profiles {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
