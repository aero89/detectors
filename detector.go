package main

import (
	"image"

	"github.com/aero89/detectors/internal/homography"
	"gocv.io/x/gocv"
)

// Person содержит результат детекции одного человека.
type Person struct {
	// BoundingBox и ImagePoint — в координатах оригинального кадра.
	BoundingBox image.Rectangle     `json:"bounding_box"`
	// ImagePoint — нижний центр bounding box (проекция ног на пол).
	ImagePoint  image.Point         `json:"image_point"`
	// RoomPoint — координаты в помещении (метры). nil если калибровка не задана.
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

// Detect возвращает список людей на кадре в координатах оригинального изображения.
//
// Pipeline:
//  1. Ресайз кадра до max_width (основное ускорение).
//  2. HOG DetectMultiScale с groupRectangles (finalThreshold).
//  3. IoU NMS — убирает оставшиеся дубли.
//  4. Масштабирование bbox обратно в оригинальные координаты.
func (d *Detector) Detect(img gocv.Mat) []Person {
	c := d.cfg

	// Масштабируем кадр вниз если шире max_width
	scale := 1.0
	working := img
	var resized gocv.Mat
	if c.MaxWidth > 0 && img.Cols() > c.MaxWidth {
		scale = float64(c.MaxWidth) / float64(img.Cols())
		newH := int(float64(img.Rows()) * scale)
		resized = gocv.NewMat()
		gocv.Resize(img, &resized, image.Point{X: c.MaxWidth, Y: newH}, 0, 0, gocv.InterpolationLinear)
		working = resized
		defer resized.Close()
	}

	winStride := image.Point{X: c.WinStrideX, Y: c.WinStrideY}
	padding := image.Point{X: c.PaddingX, Y: c.PaddingY}

	rects := d.hog.DetectMultiScaleWithParams(
		working,
		c.HitThreshold,
		winStride,
		padding,
		c.Scale,
		c.FinalThreshold,
		false,
	)

	rects = nms(rects, c.NMSThreshold)

	persons := make([]Person, 0, len(rects))
	for _, r := range rects {
		// Масштабируем bbox обратно в координаты оригинального кадра
		if scale != 1.0 {
			r = scaleRect(r, 1.0/scale)
		}
		if r.Dx() < c.MinWidth || r.Dy() < c.MinHeight {
			continue
		}
		persons = append(persons, Person{
			BoundingBox: r,
			ImagePoint:  image.Point{X: r.Min.X + r.Dx()/2, Y: r.Max.Y},
		})
	}
	return persons
}

func scaleRect(r image.Rectangle, s float64) image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: int(float64(r.Min.X) * s), Y: int(float64(r.Min.Y) * s)},
		Max: image.Point{X: int(float64(r.Max.X) * s), Y: int(float64(r.Max.Y) * s)},
	}
}

// nms — Non-Maximum Suppression по IoU.
// Из группы перекрывающихся боксов оставляет наибольший по площади.
func nms(rects []image.Rectangle, iouThreshold float64) []image.Rectangle {
	if len(rects) == 0 {
		return rects
	}
	sorted := make([]image.Rectangle, len(rects))
	copy(sorted, rects)
	sortByAreaDesc(sorted)

	kept := make([]image.Rectangle, 0, len(sorted))
	suppressed := make([]bool, len(sorted))

	for i, a := range sorted {
		if suppressed[i] {
			continue
		}
		kept = append(kept, a)
		for j := i + 1; j < len(sorted); j++ {
			if !suppressed[j] && iou(a, sorted[j]) > iouThreshold {
				suppressed[j] = true
			}
		}
	}
	return kept
}

func iou(a, b image.Rectangle) float64 {
	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	interArea := float64(inter.Dx() * inter.Dy())
	unionArea := float64(a.Dx()*a.Dy()) + float64(b.Dx()*b.Dy()) - interArea
	if unionArea <= 0 {
		return 0
	}
	return interArea / unionArea
}

func sortByAreaDesc(rects []image.Rectangle) {
	for i := 1; i < len(rects); i++ {
		key := rects[i]
		keyArea := key.Dx() * key.Dy()
		j := i - 1
		for j >= 0 && rects[j].Dx()*rects[j].Dy() < keyArea {
			rects[j+1] = rects[j]
			j--
		}
		rects[j+1] = key
	}
}
