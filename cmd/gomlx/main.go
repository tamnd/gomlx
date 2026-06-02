// SPDX-License-Identifier: Apache-2.0

// Command gomlx is a Go reimplementation of rapid-mlx: an OpenAI/Anthropic-
// compatible local LLM inference server for Apple Silicon. This is the CLI
// entry point; subcommands are wired in as each stage lands.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"text/tabwriter"

	"github.com/tamnd/gomlx/config"
	"github.com/tamnd/gomlx/engine"
	"github.com/tamnd/gomlx/models"
	"github.com/tamnd/gomlx/server"
)

const version = "0.0.1-dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd := os.Args[1]
	args := os.Args[2:]

	switch cmd {
	case "serve":
		runServe(args)
	case "models":
		runModels(args)
	case "version", "--version", "-v":
		fmt.Println(version)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "gomlx: unknown command %q\n\n", cmd)
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `gomlx - Go LLM inference server for Apple Silicon

Usage:
  gomlx <command> [flags]

Commands:
  serve     Start the inference server
  models    List available model aliases
  version   Show version
  help      Show this help

Run "gomlx <command> -h" for command flags.
`)
}

func runServe(args []string) {
	cfg := config.Default()
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	fs.StringVar(&cfg.Host, "host", cfg.Host, "bind host")
	fs.IntVar(&cfg.Port, "port", cfg.Port, "bind port")
	fs.StringVar(&cfg.ToolCallParser, "tool-call-parser", "", "tool-call parser (default: auto-detect)")
	fs.StringVar(&cfg.ReasoningParser, "reasoning-parser", "", "reasoning parser (default: auto-detect)")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", cfg.MaxTokens, "default max output tokens")
	fs.IntVar(&cfg.MaxConcurrent, "max-concurrent", cfg.MaxConcurrent, "max concurrent requests")
	fs.StringVar(&cfg.APIKey, "api-key", "", "bearer API key (default: auth disabled)")
	fs.BoolVar(&cfg.Mock, "mock", false, "run with the mock decode backend (no GPU)")
	_ = fs.Parse(args)

	if fs.NArg() > 0 {
		cfg.Model = fs.Arg(0)
	}
	if cfg.Model == "" {
		fmt.Fprintln(os.Stderr, "gomlx serve: a model alias or path is required")
		os.Exit(2)
	}

	// Resolve parsers when not explicitly set.
	det := models.DetectModelConfig(cfg.Model)
	if cfg.ToolCallParser == "" {
		cfg.ToolCallParser = det.ToolCallParser
	}
	if cfg.ReasoningParser == "" {
		cfg.ReasoningParser = det.ReasoningParser
	}

	fmt.Printf("gomlx %s\n", version)
	fmt.Printf("  model:            %s\n", cfg.Model)
	fmt.Printf("  hf_path:          %s\n", models.ResolveModel(cfg.Model))
	fmt.Printf("  tool_call_parser: %s\n", orNone(cfg.ToolCallParser))
	fmt.Printf("  reasoning_parser: %s\n", orNone(cfg.ReasoningParser))
	fmt.Printf("  is_hybrid:        %v\n", det.IsHybrid)
	fmt.Printf("  spec_decode:      %v\n", det.SupportsSpecDecode)

	// The MLX compute backend lands in stage 4. Until then, serve with the
	// mock backend so the full HTTP path runs and is benchmarkable.
	var eng engine.Engine
	if cfg.Mock {
		eng = engine.NewMockEngine(cfg.Model)
	} else {
		fmt.Fprintln(os.Stderr, "\ncompute backend not yet wired (stage 4); use --mock to run the serving layer")
		os.Exit(1)
	}

	app := server.New(cfg, eng)
	fmt.Printf("  listen:           http://%s (mock backend)\n\n", app.Addr())

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := app.ListenAndServe(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "gomlx serve: %v\n", err)
		os.Exit(1)
	}
}

func runModels(args []string) {
	fs := flag.NewFlagSet("models", flag.ExitOnError)
	_ = fs.Parse(args)

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ALIAS\tHF_PATH\tTOOL\tREASONING\tHYBRID\tMOE")
	for _, p := range models.Profiles() {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%v\t%v\n",
			p.Alias, p.HFPath, orNone(p.ToolCallParser), orNone(p.ReasoningParser), p.IsHybrid, p.IsMoE)
	}
	_ = w.Flush()
}

func orNone(s string) string {
	if s == "" {
		return "-"
	}
	return s
}
