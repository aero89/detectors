package config

import (
	"fmt"
	"log/slog"
	"os"

	"gopkg.in/yaml.v3"
	"hm-detector/internal/homography"
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
	WinStrideX int     `yaml:"win_stride_x"`
	WinStrideY int     `yaml:"win_stride_y"`
	PaddingX   int     `yaml:"padding_x"`
	PaddingY   int     `yaml:"padding_y"`
	Scale      float64 `yaml:"scale"`
	// HitThreshold — минимальный score SVM для одного окна.
	HitThreshold float64 `yaml:"hit_threshold"`
	// FinalThreshold — минимальное число перекрывающихся окон после группировки
	// (OpenCV groupRectangles). 0 = без группировки (вернёт сотни тысяч дублей!).
	FinalThreshold float64 `yaml:"final_threshold"`
	// NMSThreshold — порог IoU для NMS. Боксы с overlap > порога удаляются.
	NMSThreshold float64 `yaml:"nms_threshold"`
	// MaxWidth — максимальная ширина кадра перед детекцией (пикселей).
	// Кадр масштабируется вниз если шире. Основной рычаг скорости:
	// 640 даёт 10–15x ускорение против 1920. Bbox-ы масштабируются обратно.
	// 0 = без ресайза (не рекомендуется для кадров > 800px).
	MaxWidth int `yaml:"max_width"`

	// Коррекция bbox HOG-детектора — значения в долях от размера бокса.
	// HOG bbox обычно немного больше и смещён от реального силуэта.
	// Стандартные значения OpenCV: x=0.1, y=0.07, w=0.8, h=0.8.
	// Сдвиньте BboxXAdjust/BboxYAdjust вправо/вниз если бокс уходит влево/вверх.
	BboxXAdjust float64 `yaml:"bbox_x_adjust"` // сдвиг Min.X вправо (доля ширины)
	BboxYAdjust float64 `yaml:"bbox_y_adjust"` // сдвиг Min.Y вниз (доля высоты)
	BboxWScale  float64 `yaml:"bbox_w_scale"`  // масштаб ширины (< 1 — сужает)
	BboxHScale  float64 `yaml:"bbox_h_scale"`  // масштаб высоты (< 1 — укорачивает)

	MinWidth  int `yaml:"min_width"`
	MinHeight int `yaml:"min_height"`
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

func (c *CalibrationConfig) ToHomographyPoints() []homography.CalibrationPoint {
	out := make([]homography.CalibrationPoint, len(c.Points))
	for i, p := range c.Points {
		out[i] = homography.CalibrationPoint{
			Image: homography.Point2D{X: p.Image.X, Y: p.Image.Y},
			Room:  homography.Point2D{X: p.Room.X, Y: p.Room.Y},
		}
	}
	return out
}

// LoadConfig загружает конфиг из файла.
// Приоритет пути: флаг -config → переменная окружения CONFIG_PATH → "config.yaml".
// Если файл не найден — возвращает конфиг с дефолтными значениями (без калибровки).
func LoadConfig(flagPath string) (*Config, error) {
	path := resolveConfigPath(flagPath)

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		var cfg Config
		cfg.applyDefaults()
		slog.Warn("config file not found, using defaults", "path", path)
		return &cfg, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}
	cfg.applyDefaults()
	slog.Info("config loaded", "path", path)
	return &cfg, nil
}

// resolveConfigPath определяет путь к конфигу:
// 1. Флаг -config (если не дефолтный "config.yaml", значит задан явно)
// 2. CONFIG_PATH из окружения
// 3. "config.yaml" по умолчанию
func resolveConfigPath(flagPath string) string {
	if flagPath != "config.yaml" {
		// Флаг был задан явно
		return flagPath
	}
	if env := os.Getenv("CONFIG_PATH"); env != "" {
		return env
	}
	return flagPath
}

func (c *Config) applyDefaults() {
	if c.Server.Addr == "" {
		c.Server.Addr = ":8080"
	}
	d := &c.Detector
	if d.WinStrideX == 0 {
		d.WinStrideX = 16
	}
	if d.WinStrideY == 0 {
		d.WinStrideY = 16
	}
	if d.Scale == 0 {
		d.Scale = 1.05
	}
	if d.FinalThreshold == 0 {
		d.FinalThreshold = 2
	}
	if d.NMSThreshold == 0 {
		d.NMSThreshold = 0.65
	}
	if d.MaxWidth == 0 {
		d.MaxWidth = 640
	}
	if d.BboxXAdjust == 0 {
		d.BboxXAdjust = 0.1
	}
	if d.BboxYAdjust == 0 {
		d.BboxYAdjust = 0.07
	}
	if d.BboxWScale == 0 {
		d.BboxWScale = 0.8
	}
	if d.BboxHScale == 0 {
		d.BboxHScale = 0.8
	}
	if d.MinWidth == 0 {
		d.MinWidth = 48
	}
	if d.MinHeight == 0 {
		d.MinHeight = 96
	}
}
