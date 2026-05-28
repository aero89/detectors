package main

import (
	"log/slog"
	"os"
	"flag"

	"github.com/aero89/detectors/internal/homography"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

func main() {
	cfgPath := flag.String("config", "config.yaml", "path to config file")
	flag.Parse()

	cfg, err := LoadConfig(*cfgPath)
	if err != nil {
		slog.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	warnIfMaxWidthTooSmall(cfg.Detector)

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

	app := fiber.New(fiber.Config{
		AppName: "detectors",
	})
	app.Use(recover.New())
	app.Use(logger.New())

	app.Post("/detect", dh.Detect)
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusOK)
	})

	slog.Info("detector ready",
		"max_width", cfg.Detector.MaxWidth,
		"win_stride", cfg.Detector.WinStrideX,
		"scale", cfg.Detector.Scale)
	slog.Info("server starting", "addr", cfg.Server.Addr)
	if err := app.Listen(cfg.Server.Addr); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}

// warnIfMaxWidthTooSmall предупреждает если max_width настолько мал,
// что типичный человек станет меньше минимального окна HOG (64×128px).
// HOG не может детектировать объекты меньше своего окна.
func warnIfMaxWidthTooSmall(cfg DetectorConfig) {
	// HOG минимальная высота окна — 128px
	const hogMinHeight = 128
	if cfg.MaxWidth == 0 {
		return
	}
	// Оцениваем: если min_height человека в оригинале — cfg.MinHeight,
	// то на working-кадре он будет ещё меньше при ресайзе.
	// Например: оригинал 1920px → working 640px → scale=0.333
	// Человек 300px высотой → 100px на working. Это > 128? Нет → предупреждение.
	//
	// Мы не знаем размер оригинального кадра заранее, поэтому проверяем
	// min_height как нижнюю границу: если min_height < hogMinHeight,
	// детектор будет отфильтровывать корректные детекции на working-кадре.
	if cfg.MinHeight < hogMinHeight {
		slog.Warn("min_height is below HOG window height; detections may be filtered",
			"min_height", cfg.MinHeight,
			"hog_min_window_height", hogMinHeight)
	}
	if cfg.MaxWidth < 960 {
		slog.Warn("max_width may be too small for 1080p input: people could fall below the 64×128 HOG window",
			"max_width", cfg.MaxWidth,
			"recommended_min", 1280)
	}
}
