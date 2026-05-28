package detector

import (
	"fmt"
	"image"
	"log/slog"

	"gocv.io/x/gocv"
	"hm-detector/internal/config"
	"hm-detector/internal/homography"
)

// Person содержит результат детекции одного человека.
type Person struct {
	BoundingBox image.Rectangle     `json:"bounding_box"`
	ImagePoint  image.Point         `json:"image_point"`
	RoomPoint   *homography.Point2D `json:"room_point,omitempty"`
}

// DetectResult — результат детекции с опциональным дебагом.
type DetectResult struct {
	Persons []Person   `json:"persons"`
	Debug   *DebugInfo `json:"debug,omitempty"`
}

// DebugInfo — промежуточные данные для диагностики.
type DebugInfo struct {
	WorkingSize  image.Point    `json:"working_size"`
	WorkingScale float64        `json:"working_scale"`
	Method       string         `json:"method"`
	Extra        map[string]any `json:"extra,omitempty"`
}

// Detector — общий интерфейс детектора.
type Detector interface {
	Detect(imgBytes []byte, withDebug bool) (DetectResult, error)
	Reset()
	Close()
}

// New создаёт детектор согласно конфигу.
func New(cfg config.DetectorConfig) (Detector, error) {
	switch cfg.Method {
	case "dnn":
		slog.Info("using DNN detector", "model", cfg.DNN.Model)
		return newDNNDetector(cfg)
	case "backsub":
		slog.Info("using BackSub (MOG2) detector")
		return newBackSubDetector(cfg), nil
	default:
		return nil, fmt.Errorf("unknown detector method %q (expected: dnn, backsub)", cfg.Method)
	}
}

// DecodeImage декодирует байты изображения в gocv.Mat.
func DecodeImage(imgBytes []byte) (gocv.Mat, error) {
	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to decode image")
	}
	return mat, nil
}

// resizeToMaxWidth масштабирует изображение если оно шире maxWidth.
// Возвращает scale (working/original), рабочий Mat и функцию освобождения.
func resizeToMaxWidth(img gocv.Mat, maxWidth int) (scale float64, working gocv.Mat, cleanup func()) {
	if maxWidth <= 0 || img.Cols() <= maxWidth {
		return 1.0, img, func() {}
	}
	scale = float64(maxWidth) / float64(img.Cols())
	newH := int(float64(img.Rows()) * scale)
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Point{X: maxWidth, Y: newH}, 0, 0, gocv.InterpolationLinear)
	return scale, resized, func() { resized.Close() }
}

func scaleRect(r image.Rectangle, s float64) image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: int(float64(r.Min.X) * s), Y: int(float64(r.Min.Y) * s)},
		Max: image.Point{X: int(float64(r.Max.X) * s), Y: int(float64(r.Max.Y) * s)},
	}
}

func footPoint(r image.Rectangle) image.Point {
	return image.Point{X: r.Min.X + r.Dx()/2, Y: r.Max.Y}
}
