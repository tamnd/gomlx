// SPDX-License-Identifier: Apache-2.0

// Command gomlx is an OpenAI/Anthropic-compatible local LLM inference server
// for Apple Silicon. This is the CLI entry point; subcommands are wired in as
// each stage lands.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"text/tabwriter"

	"github.com/tamnd/gomlx/bench"
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
	case "bench":
		runBench(args)
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
  bench     Load-test an OpenAI-compatible endpoint
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

	// The mock backend exercises the full HTTP path with no GPU. The real path
	// loads a local model directory through the MLX engine, which needs a binary
	// built with the mlx tag and an MLX runtime.
	var eng engine.Engine
	backend := "mock backend"
	if cfg.Mock {
		eng = engine.NewMockEngine(cfg.Model)
	} else {
		dir := cfg.Model
		if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
			fmt.Fprintf(os.Stderr, "\ngomlx serve: %q is not a local model directory.\n"+
				"Pass a directory holding config.json, tokenizer.json, and model.safetensors,\n"+
				"or use --mock to run the serving layer without a model.\n", cfg.Model)
			os.Exit(1)
		}
		mlxEng, err := engine.NewMLXEngine(cfg.Model, dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\ngomlx serve: %v\n"+
				"Build with -tags mlx and an MLX runtime, or use --mock.\n", err)
			os.Exit(1)
		}
		eng = mlxEng
		backend = "mlx backend"
	}

	app := server.New(cfg, eng)
	fmt.Printf("  listen:           http://%s (%s)\n\n", app.Addr(), backend)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := app.ListenAndServe(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "gomlx serve: %v\n", err)
		os.Exit(1)
	}
}

func runBench(args []string) {
	var cfg bench.Config
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	fs.StringVar(&cfg.URL, "url", "http://127.0.0.1:8000", "base URL of the server")
	fs.StringVar(&cfg.Model, "model", "", "model name to send in the request")
	fs.StringVar(&cfg.APIKey, "api-key", "", "bearer API key, if the server requires one")
	fs.StringVar(&cfg.Prompt, "prompt", "Write a short paragraph about the sea.", "prompt to send")
	fs.IntVar(&cfg.MaxTokens, "max-tokens", 128, "max output tokens per request")
	fs.Float64Var(&cfg.Temperature, "temperature", 0, "sampling temperature")
	fs.IntVar(&cfg.Concurrency, "concurrency", 1, "number of concurrent clients")
	fs.IntVar(&cfg.Requests, "requests", 0, "total requests (default: one per client)")
	fs.BoolVar(&cfg.Stream, "stream", false, "stream responses to measure time to first token")
	_ = fs.Parse(args)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Printf("benchmarking %s (%d clients, %d requests, stream=%v)\n\n",
		cfg.URL, cfg.Concurrency, max(cfg.Requests, cfg.Concurrency), cfg.Stream)
	rep, err := bench.Run(ctx, cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "gomlx bench: %v\n", err)
		os.Exit(1)
	}
	fmt.Print(rep.String())
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
