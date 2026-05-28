package main

import (
	"image"

	"github.com/aero89/detectors/internal/homography"
)

// Person содержит результат детекции одного человека.
type Person struct {
	BoundingBox image.Rectangle     `json:"bounding_box"`
	// ImagePoint — нижний центр bbox (проекция ног на пол).
	ImagePoint  image.Point         `json:"image_point"`
	// RoomPoint — координаты в помещении (метры). nil если калибровка не задана.
	RoomPoint   *homography.Point2D `json:"room_point,omitempty"`
}

// DetectResult — результат детекции с опциональным дебагом.
type DetectResult struct {
	Persons []Person   `json:"persons"`
	Debug   *DebugInfo `json:"debug,omitempty"`
}

// DebugInfo — промежуточные данные для диагностики.
type DebugInfo struct {
	WorkingSize  image.Point `json:"working_size"`
	WorkingScale float64     `json:"working_scale"`
	// Method — каким детектором обработан кадр
	Method string `json:"method"`
	// Extra — дополнительные поля, специфичные для метода
	Extra map[string]any `json:"extra,omitempty"`
}

// PersonDetector — общий интерфейс детектора.
type PersonDetector interface {
	Detect(img []byte, withDebug bool) (DetectResult, error)
	Reset()
	Close()
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
