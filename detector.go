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

// DetectResult — полный результат детекции, включая опциональный дебаг.
type DetectResult struct {
	Persons []Person    `json:"persons"`
	Debug   *DebugInfo  `json:"debug,omitempty"`
}

// DebugInfo показывает промежуточные данные каждого шага пайплайна.
// Используйте ?debug=true чтобы получить его в ответе.
type DebugInfo struct {
	// Размер кадра, поданного в HOG (после ресайза)
	WorkingSize image.Point `json:"working_size"`
	// Коэффициент масштабирования: working / original
	// 1.0 = ресайз не применялся
	WorkingScale float64 `json:"working_scale"`
	// Сырые боксы от HOG — в координатах working (до любых поправок)
	RawRectsWorking []image.Rectangle `json:"raw_rects_working"`
	// После NMS, в координатах working (до bbox-коррекции и scale-back)
	AfterNMSWorking []image.Rectangle `json:"after_nms_working"`
	// После bbox-коррекции, в координатах working
	AfterAdjustWorking []image.Rectangle `json:"after_adjust_working"`
	// Итоговые боксы в координатах оригинального кадра
	FinalRectsOriginal []image.Rectangle `json:"final_rects_original"`
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

// Detect запускает пайплайн детекции.
// withDebug=true заполняет DetectResult.Debug промежуточными данными каждого шага.
func (d *Detector) Detect(img gocv.Mat, withDebug bool) DetectResult {
	c := d.cfg

	// Шаг 1: ресайз до max_width
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

	workingSize := image.Point{X: working.Cols(), Y: working.Rows()}

	// Шаг 2: HOG
	winStride := image.Point{X: c.WinStrideX, Y: c.WinStrideY}
	padding := image.Point{X: c.PaddingX, Y: c.PaddingY}
	rawRects := d.hog.DetectMultiScaleWithParams(
		working,
		c.HitThreshold,
		winStride,
		padding,
		c.Scale,
		c.FinalThreshold,
		false,
	)

	// Шаг 3: NMS (в working-координатах)
	afterNMS := nms(rawRects, c.NMSThreshold)

	// Шаг 4: bbox-коррекция + scale-back
	afterAdjust := make([]image.Rectangle, len(afterNMS))
	finalRects := make([]image.Rectangle, 0, len(afterNMS))
	persons := make([]Person, 0, len(afterNMS))

	for i, r := range afterNMS {
		adjusted := adjustRect(r, c.BboxXAdjust, c.BboxYAdjust, c.BboxWScale, c.BboxHScale)
		afterAdjust[i] = adjusted

		orig := adjusted
		if scale != 1.0 {
			orig = scaleRect(adjusted, 1.0/scale)
		}
		if orig.Dx() < c.MinWidth || orig.Dy() < c.MinHeight {
			continue
		}
		finalRects = append(finalRects, orig)
		persons = append(persons, Person{
			BoundingBox: orig,
			ImagePoint:  image.Point{X: orig.Min.X + orig.Dx()/2, Y: orig.Max.Y},
		})
	}

	result := DetectResult{Persons: persons}
	if withDebug {
		result.Debug = &DebugInfo{
			WorkingSize:        workingSize,
			WorkingScale:       scale,
			RawRectsWorking:    rawRects,
			AfterNMSWorking:    afterNMS,
			AfterAdjustWorking: afterAdjust,
			FinalRectsOriginal: finalRects,
		}
	}
	return result
}

// adjustRect корректирует bbox HOG-детектора.
// xAdj/yAdj — сдвиг левого верхнего угла в долях ширины/высоты.
// wScale/hScale — масштаб размеров.
func adjustRect(r image.Rectangle, xAdj, yAdj, wScale, hScale float64) image.Rectangle {
	w := float64(r.Dx())
	h := float64(r.Dy())
	x := float64(r.Min.X) + xAdj*w
	y := float64(r.Min.Y) + yAdj*h
	return image.Rectangle{
		Min: image.Point{X: int(x), Y: int(y)},
		Max: image.Point{X: int(x + w*wScale), Y: int(y + h*hScale)},
	}
}

func scaleRect(r image.Rectangle, s float64) image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: int(float64(r.Min.X) * s), Y: int(float64(r.Min.Y) * s)},
		Max: image.Point{X: int(float64(r.Max.X) * s), Y: int(float64(r.Max.Y) * s)},
	}
}

// nms — Non-Maximum Suppression по IoU.
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
