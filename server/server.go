// SPDX-License-Identifier: Apache-2.0

// Package server wires the HTTP application: route registration, middleware,
// and lifespan (engine start/stop).
package server

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/tamnd/gomlx/config"
	"github.com/tamnd/gomlx/engine"
	"github.com/tamnd/gomlx/middleware"
	"github.com/tamnd/gomlx/routes"
)

// App is the assembled HTTP server with its engine and config.
type App struct {
	cfg    config.ServerConfig
	engine engine.Engine
	deps   *routes.Deps
}

// New builds an App for the given config and engine.
func New(cfg config.ServerConfig, eng engine.Engine) *App {
	return &App{
		cfg:    cfg,
		engine: eng,
		deps: &routes.Deps{
			Engine:           eng,
			Model:            cfg.Model,
			DefaultMaxTokens: cfg.MaxTokens,
			ToolCallParser:   cfg.ToolCallParser,
		},
	}
}

// Handler returns the fully-wired http.Handler with middleware applied.
func (a *App) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", a.deps.ChatCompletions)
	mux.HandleFunc("POST /v1/completions", a.deps.Completions)
	mux.HandleFunc("GET /v1/models", a.deps.Models)
	mux.HandleFunc("GET /health", a.deps.Health)
	mux.HandleFunc("GET /v1/health", a.deps.Health)

	return middleware.Chain(mux,
		middleware.CORS,
		middleware.Auth(a.cfg.APIKey),
	)
}

// ListenAndServe starts the engine, serves until ctx is cancelled, then stops
// the engine. It blocks until shutdown completes.
func (a *App) ListenAndServe(ctx context.Context) error {
	if err := a.engine.Start(ctx); err != nil {
		return err
	}
	defer func() {
		stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = a.engine.Stop(stopCtx)
	}()

	srv := &http.Server{
		Addr:    a.cfg.Host + ":" + strconv.Itoa(a.cfg.Port),
		Handler: a.Handler(),
	}

	errCh := make(chan error, 1)
	go func() { errCh <- srv.ListenAndServe() }()

	select {
	case <-ctx.Done():
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shutCtx)
	case err := <-errCh:
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}
}

// Addr returns the configured listen address.
func (a *App) Addr() string {
	return a.cfg.Host + ":" + strconv.Itoa(a.cfg.Port)
}
