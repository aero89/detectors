package main

import (
	"fmt"
	"os"

	"github.com/aero89/detectors/internal/homography"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Server      ServerConfig      `yaml:"server"`
	Detector    DetectorConfig    `yaml:"detector"`
	Calibration CalibrationConfig `yaml:"calibration"`
}

type ServerConfig struct {
	Addr string `yaml:"addr"`
}

type DetectorConfig struct {
	WinStrideX   int     `yaml:"win_stride_x"`
	WinStrideY   int     `yaml:"win_stride_y"`
	PaddingX     int     `yaml:"padding_x"`
	PaddingY     int     `yaml:"padding_y"`
	Scale        float64 `yaml:"scale"`
	HitThreshold float64 `yaml:"hit_threshold"`
	MinWidth     int     `yaml:"min_width"`
	MinHeight    int     `yaml:"min_height"`
}

// CalibrationConfig задаёт соответствие точек изображения и помещения.
// Нужно минимум 4 пары для вычисления гомографии.
//
// Как снять калибровку:
//  1. Разместьте на полу 4+ маркеров (не на одной прямой).
//  2. Измерьте их координаты в метрах.
//  3. Определите пиксельные координаты тех же маркеров на кадре.
type CalibrationConfig struct {
	Points []calibPoint `yaml:"points"`
}

type calibPoint struct {
	Image struct {
		X float64 `yaml:"x"`
		Y float64 `yaml:"y"`
	} `yaml:"image"`
	Room struct {
		X float64 `yaml:"x"`
		Y float64 `yaml:"y"`
	} `yaml:"room"`
}

func (c *CalibrationConfig) toHomographyPoints() []homography.CalibrationPoint {
	out := make([]homography.CalibrationPoint, len(c.Points))
	for i, p := range c.Points {
		out[i] = homography.CalibrationPoint{
			Image: homography.Point2D{X: p.Image.X, Y: p.Image.Y},
			Room:  homography.Point2D{X: p.Room.X, Y: p.Room.Y},
		}
	}
	return out
}

func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}
	cfg.applyDefaults()
	return &cfg, nil
}

func (c *Config) applyDefaults() {
	if c.Server.Addr == "" {
		c.Server.Addr = ":8080"
	}
	d := &c.Detector
	if d.WinStrideX == 0 {
		d.WinStrideX = 8
	}
	if d.WinStrideY == 0 {
		d.WinStrideY = 8
	}
	if d.Scale == 0 {
		d.Scale = 1.05
	}
	if d.MinWidth == 0 {
		d.MinWidth = 48
	}
	if d.MinHeight == 0 {
		d.MinHeight = 96
	}
}
