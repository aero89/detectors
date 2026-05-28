package main

import (
	"log/slog"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"hm-detector/internal/config"
	"hm-detector/internal/controller"
	"hm-detector/internal/detector"
	"hm-detector/internal/homography"
)

func main() {
	cfg, err := config.LoadConfig("/app/config.yaml")
	if err != nil {
		slog.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	det, err := detector.New(cfg.Detector)
	if err != nil {
		slog.Error("failed to init detector", "err", err)
		os.Exit(1)
	}
	defer det.Close()

	var hom *homography.Homography
	if pts := cfg.Calibration.ToHomographyPoints(); len(pts) >= 4 {
		hom, err = homography.New(pts)
		if err != nil {
			slog.Error("failed to compute homography", "err", err)
			os.Exit(1)
		}
		slog.Info("homography ready", "calibration_points", len(pts))
	} else {
		slog.Warn("homography disabled: need >= 4 calibration points; room_point will be null")
	}

	dh := &controller.DetectHandler{Detector: det, Homography: hom}

	app := fiber.New(fiber.Config{AppName: "detectors"})
	app.Use(recover.New())
	app.Use(logger.New())

	app.Post("/detect", dh.Detect)
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusOK)
	})
	// Сброс модели фона (актуально для method: backsub)
	app.Post("/reset", func(c *fiber.Ctx) error {
		det.Reset()
		slog.Info("detector reset")
		return c.JSON(fiber.Map{"status": "ok"})
	})

	slog.Info("server starting", "addr", cfg.Server.Addr, "method", cfg.Detector.Method)
	if err := app.Listen(cfg.Server.Addr); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}
