package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aero89/detectors/internal/homography"
)

func main() {
	cfgPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*cfgPath)
	if err != nil {
		slog.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	det := NewDetector(cfg.Detector)
	defer det.Close()

	var hom *homography.Homography
	if pts := cfg.Calibration.toHomographyPoints(); len(pts) >= 4 {
		hom, err = homography.New(pts)
		if err != nil {
			slog.Error("failed to compute homography", "err", err)
			os.Exit(1)
		}
		slog.Info("homography ready", "calibration_points", len(pts))
	} else {
		slog.Warn("homography disabled: need >= 4 calibration points; room_point will be null")
	}

	dh := &DetectHandler{detector: det, homography: hom}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /detect", dh.Detect)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	srv := &http.Server{
		Addr:         cfg.Server.Addr,
		Handler:      mux,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
	}

	go func() {
		slog.Info("server started", "addr", cfg.Server.Addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("shutdown error", "err", err)
	}
}
