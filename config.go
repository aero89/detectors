package main

import (
	"fmt"
	"log/slog"
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
	// MaxWidth — ширина кадра для детекции (0 = без ресайза).
	// Bbox-ы масштабируются обратно в оригинальные координаты автоматически.
	MaxWidth int `yaml:"max_width"`

	// BackSub — параметры вычитания фона (MOG2).
	// Рекомендуется для статичной камеры в помещении: работает с любым углом,
	// не зависит от формы силуэта человека.
	BackSub BackSubConfig `yaml:"back_sub"`

	// Contour — фильтрация найденных контуров переднего плана.
	Contour ContourConfig `yaml:"contour"`
}

// BackSubConfig — параметры BackgroundSubtractorMOG2.
type BackSubConfig struct {
	// History — количество кадров для построения модели фона.
	// Больше → стабильнее фон, но дольше инициализация.
	History int `yaml:"history"`
	// VarThreshold — порог дисперсии. Меньше → чувствительнее, больше шума.
	VarThreshold float64 `yaml:"var_threshold"`
	// DetectShadows — классифицировать тени отдельно (замедляет, обычно не нужно).
	DetectShadows bool `yaml:"detect_shadows"`

	// MorphErodeSize — размер ядра эрозии (убирает мелкий шум). 0 = выключено.
	MorphErodeSize int `yaml:"morph_erode_size"`
	// MorphDilateSize — размер ядра дилатации (заполняет дыры в маске человека).
	MorphDilateSize int `yaml:"morph_dilate_size"`
}

// ContourConfig — правила фильтрации контуров переднего плана.
type ContourConfig struct {
	// MinArea / MaxArea — площадь контура в пикселях рабочего кадра.
	// Слишком маленькие — шум; слишком большие — группа людей или артефакт.
	MinArea float64 `yaml:"min_area"`
	MaxArea float64 `yaml:"max_area"`

	// MinWidth / MinHeight — минимальный размер bbox контура в пикселях
	// оригинального кадра (после масштабирования обратно).
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

// LoadConfig загружает конфиг из файла.
// Приоритет пути: флаг -config → CONFIG_PATH → "config.yaml".
// Если файл не найден — запускается с дефолтами (без калибровки).
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

func resolveConfigPath(flagPath string) string {
	if flagPath != "config.yaml" {
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
	if d.MaxWidth == 0 {
		d.MaxWidth = 960
	}
	bs := &d.BackSub
	if bs.History == 0 {
		bs.History = 500
	}
	if bs.VarThreshold == 0 {
		bs.VarThreshold = 25
	}
	if bs.MorphErodeSize == 0 {
		bs.MorphErodeSize = 3
	}
	if bs.MorphDilateSize == 0 {
		bs.MorphDilateSize = 15
	}
	ct := &d.Contour
	if ct.MinArea == 0 {
		ct.MinArea = 500
	}
	if ct.MaxArea == 0 {
		ct.MaxArea = 80000
	}
	if ct.MinWidth == 0 {
		ct.MinWidth = 30
	}
	if ct.MinHeight == 0 {
		ct.MinHeight = 30
	}
}
