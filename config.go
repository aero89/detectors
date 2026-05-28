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
	// Method — алгоритм детекции: "dnn" или "backsub".
	// "dnn"     — нейросеть (YOLO/MobileNet). Работает на одном кадре,
	//             не зависит от угла камеры. Требует файлы модели.
	// "backsub" — вычитание фона (MOG2). Только движущиеся объекты,
	//             не требует модели. Хорош для видеопотока.
	Method string `yaml:"method"`

	// MaxWidth — ширина рабочего кадра (0 = без ресайза).
	MaxWidth int `yaml:"max_width"`

	DNN     DNNConfig     `yaml:"dnn"`
	BackSub BackSubConfig `yaml:"back_sub"`
	Contour ContourConfig `yaml:"contour"`
}

// DNNConfig — параметры нейросетевого детектора.
type DNNConfig struct {
	// Model — путь к файлу весов:
	//   YOLO Darknet: yolov4-tiny.weights
	//   ONNX:         yolov8n.onnx
	Model string `yaml:"model"`
	// Config — путь к конфигу сети (только для Darknet .cfg; для ONNX не нужен).
	Config string `yaml:"config"`
	// Classes — файл с именами классов (по одному на строку, как coco.names).
	Classes string `yaml:"classes"`

	// InputSize — размер входа сети. Для YOLO обычно 416 или 640.
	InputWidth  int `yaml:"input_width"`
	InputHeight int `yaml:"input_height"`

	// ConfThreshold — минимальная уверенность детекции (0–1).
	ConfThreshold float64 `yaml:"conf_threshold"`
	// NMSThreshold — порог IoU для NMS внутри DNN-постобработки.
	NMSThreshold float64 `yaml:"nms_threshold"`

	// PersonClassID — ID класса «человек» в модели.
	// В COCO (YOLOv4, YOLOv8): 0.
	PersonClassID int `yaml:"person_class_id"`

	// Backend / Target — ускоритель вывода.
	// Backend: 0=default, 1=halide, 2=openvino, 3=opencv, 4=vulkan, 5=cuda
	// Target:  0=cpu, 1=opencl, 2=opencl_fp16, 3=myriad, 6=cuda, 7=cuda_fp16
	Backend int `yaml:"backend"`
	Target  int `yaml:"target"`
}

// BackSubConfig — параметры MOG2.
type BackSubConfig struct {
	History       int     `yaml:"history"`
	VarThreshold  float64 `yaml:"var_threshold"`
	DetectShadows bool    `yaml:"detect_shadows"`

	MorphErodeSize  int `yaml:"morph_erode_size"`
	MorphDilateSize int `yaml:"morph_dilate_size"`
}

// ContourConfig — фильтрация контуров (используется обоими методами).
type ContourConfig struct {
	MinArea   float64 `yaml:"min_area"`
	MaxArea   float64 `yaml:"max_area"`
	MinWidth  int     `yaml:"min_width"`
	MinHeight int     `yaml:"min_height"`
}

// CalibrationConfig задаёт соответствие точек изображения и помещения.
// Нужно минимум 4 пары для вычисления гомографии.
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
	if d.Method == "" {
		d.Method = "backsub"
	}
	if d.MaxWidth == 0 {
		d.MaxWidth = 960
	}

	dn := &d.DNN
	if dn.InputWidth == 0 {
		dn.InputWidth = 416
	}
	if dn.InputHeight == 0 {
		dn.InputHeight = 416
	}
	if dn.ConfThreshold == 0 {
		dn.ConfThreshold = 0.5
	}
	if dn.NMSThreshold == 0 {
		dn.NMSThreshold = 0.4
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
