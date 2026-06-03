// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

// The stdio transport is how a session reaches a server that runs as a local
// subprocess: the server's stdin and stdout become the JSON-RPC byte stream. This
// file launches that process, presents its pipes as one stream the connection can
// read and write, and shuts it down cleanly when the session closes, falling back
// to a kill if the process will not exit on its own.

// stdioStreamKillGrace is how long a process is given to exit after its stdin is
// closed before it is killed.
const stdioStreamKillGrace = 2 * time.Second

// stdioStream couples a subprocess's stdin and stdout into one
// io.ReadWriteCloser. Reads come from the process's stdout, writes go to its
// stdin, and Close shuts the process down.
type stdioStream struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser

	closeOnce sync.Once
	closeErr  error
}

func (s *stdioStream) Read(p []byte) (int, error)  { return s.stdout.Read(p) }
func (s *stdioStream) Write(p []byte) (int, error) { return s.stdin.Write(p) }

// Close ends the conversation by closing the process's stdin, then waits briefly
// for it to exit and kills it if it does not. Closing stdin is the protocol's own
// signal to a well-behaved server that it should shut down.
func (s *stdioStream) Close() error {
	s.closeOnce.Do(func() {
		_ = s.stdin.Close()

		done := make(chan error, 1)
		go func() { done <- s.cmd.Wait() }()

		timer := time.NewTimer(stdioStreamKillGrace)
		defer timer.Stop()
		select {
		case err := <-done:
			s.closeErr = err
		case <-timer.C:
			_ = s.cmd.Process.Kill()
			s.closeErr = <-done
		}
	})
	return s.closeErr
}

// startStdio launches the server described by cfg and returns its coupled stream.
// The child inherits the current environment with cfg.Env applied on top, so a
// server can be handed a token or a path without losing PATH and the rest.
func startStdio(ctx context.Context, cfg ServerConfig) (*stdioStream, error) {
	if cfg.Command == "" {
		return nil, fmt.Errorf("mcp: stdio server %q has no command", cfg.Name)
	}
	cmd := exec.CommandContext(ctx, cfg.Command, cfg.Args...)
	cmd.Env = mergeEnv(os.Environ(), cfg.Env)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdio server %q stdin: %w", cfg.Name, err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdio server %q stdout: %w", cfg.Name, err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: start stdio server %q: %w", cfg.Name, err)
	}
	return &stdioStream{cmd: cmd, stdin: stdin, stdout: stdout}, nil
}

// DialStdio launches a stdio MCP server, completes the initialize handshake, and
// returns a ready session. Closing the session shuts the subprocess down. The
// passed context bounds the handshake and is also tied to the process lifetime,
// so cancelling it terminates the server.
func DialStdio(ctx context.Context, cfg ServerConfig, info ClientInfo) (*Session, error) {
	stream, err := startStdio(ctx, cfg)
	if err != nil {
		return nil, err
	}
	conn := NewConn(stream, nil)
	sess := NewSession(conn, cfg.Name, info)
	if err := sess.Initialize(ctx); err != nil {
		_ = sess.Close()
		return nil, err
	}
	return sess, nil
}

// mergeEnv returns base with the overrides applied, replacing a base entry when a
// key collides and appending the rest. The result is sorted for determinism.
func mergeEnv(base []string, overrides map[string]string) []string {
	if len(overrides) == 0 {
		return base
	}
	merged := make(map[string]string, len(base)+len(overrides))
	for _, kv := range base {
		if k, v, ok := strings.Cut(kv, "="); ok {
			merged[k] = v
		}
	}
	maps.Copy(merged, overrides)
	out := make([]string, 0, len(merged))
	for k, v := range merged {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}
