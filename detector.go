package main

import (
	"image"

	"github.com/aero89/detectors/internal/homography"
	"gocv.io/x/gocv"
)

// Person содержит результат детекции одного человека.
type Person struct {
	BoundingBox image.Rectangle     `json:"bounding_box"`
	// ImagePoint — нижний центр bounding box в пикселях.
	// Это проекция ног человека на пол, которая отображается гомографией.
	ImagePoint  image.Point         `json:"image_point"`
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
// Двухступенчатая фильтрация:
//  1. OpenCV groupRectangles (finalThreshold) — убирает одиночные окна без поддержки.
//  2. IoU NMS (nmsThreshold) — убирает дубли после группировки.
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
		c.FinalThreshold,
		false,
	)

	rects = nms(rects, c.NMSThreshold)

	persons := make([]Person, 0, len(rects))
	for _, r := range rects {
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

// nms выполняет Non-Maximum Suppression по IoU.
// Из группы перекрывающихся боксов (IoU > threshold) оставляет наибольший по площади.
func nms(rects []image.Rectangle, iouThreshold float64) []image.Rectangle {
	if len(rects) == 0 {
		return rects
	}

	// Сортируем по убыванию площади: жадно берём самый крупный бокс первым
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
	// Insertion sort — список после groupRectangles обычно небольшой
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
