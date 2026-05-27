package main

import (
	"image"

	"github.com/aero89/detectors/internal/homography"
	"gocv.io/x/gocv"
)

// Person содержит результат детекции одного человека.
type Person struct {
	BoundingBox image.Rectangle   `json:"bounding_box"`
	// ImagePoint — нижний центр bounding box в пикселях.
	// Это проекция ног человека на пол, которая затем отображается гомографией.
	ImagePoint  image.Point        `json:"image_point"`
	// RoomPoint — координаты в помещении (метры от начала отсчёта).
	// nil если калибровка не задана.
	RoomPoint   *homography.Point2D `json:"room_point,omitempty"`
}

// Detector детектирует людей с помощью HOG + LinearSVM.
type Detector struct {
	hog gocv.HOGDescriptor
	cfg DetectorConfig
}

func NewDetector(cfg DetectorConfig) *Detector {
	hog := gocv.NewHOGDescriptor()
	hog.SetSVMDetector(gocv.HOGDefaultPeopleDetector())
	return &Detector{hog: hog, cfg: cfg}
}

func (d *Detector) Close() {
	d.hog.Close()
}

// Detect возвращает список людей на кадре.
func (d *Detector) Detect(img gocv.Mat) []Person {
	c := d.cfg
	winStride := image.Point{X: c.WinStrideX, Y: c.WinStrideY}
	padding := image.Point{X: c.PaddingX, Y: c.PaddingY}

	rects := d.hog.DetectMultiScaleWithParams(
		img,
		c.HitThreshold,
		winStride,
		padding,
		c.Scale,
		0,
		false,
	)

	persons := make([]Person, 0, len(rects))
	for _, r := range rects {
		if r.Dx() < c.MinWidth || r.Dy() < c.MinHeight {
			continue
		}
		footPoint := image.Point{
			X: r.Min.X + r.Dx()/2,
			Y: r.Max.Y,
		}
		persons = append(persons, Person{
			BoundingBox: r,
			ImagePoint:  footPoint,
		})
	}
	return persons
}
