package main

import (
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"hm-detector/internal/config"
	"hm-detector/internal/controller"
	"hm-detector/internal/detector"
	"hm-detector/internal/homography"
	"log/slog"
	"os"
)

func main() {
	//cfgPath := flag.String("config", "config.yaml", "path to config file")
	//flag.Parse()

	cfg, err := config.LoadConfig("/app/config.yaml")
	if err != nil {
		slog.Error("failed to load config", "err", err)
		os.Exit(1)
	}

	det := detector.NewDetector(cfg.Detector)
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

	app := fiber.New(fiber.Config{
		AppName: "detectors",
	})
	app.Use(recover.New())
	app.Use(logger.New())

	app.Post("/detect", dh.Detect)
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.SendStatus(fiber.StatusOK)
	})

	slog.Info("server starting", "addr", cfg.Server.Addr)
	if err := app.Listen(cfg.Server.Addr); err != nil {
		slog.Error("server error", "err", err)
		os.Exit(1)
	}
}
