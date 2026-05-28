package main

import (
	"image"
	"image/color"
	"sync"

	"github.com/aero89/detectors/internal/homography"
	"gocv.io/x/gocv"
)

// Person содержит результат детекции одного человека.
type Person struct {
	// BoundingBox и ImagePoint — в координатах оригинального кадра.
	BoundingBox image.Rectangle `json:"bounding_box"`
	// ImagePoint — нижний центр bbox (проекция ног/основания на пол).
	ImagePoint image.Point `json:"image_point"`
	// RoomPoint — координаты в помещении (метры). nil если калибровка не задана.
	RoomPoint *homography.Point2D `json:"room_point,omitempty"`
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
	// ForegroundPixels — количество пикселей переднего плана после морфологии
	ForegroundPixels int `json:"foreground_pixels"`
	// RawContours — bbox каждого контура до фильтрации (в working-координатах)
	RawContours []image.Rectangle `json:"raw_contours_working"`
	// FinalRects — итоговые bbox в оригинальных координатах
	FinalRects []image.Rectangle `json:"final_rects_original"`
}

// Detector — детектор людей на основе вычитания фона (MOG2).
// Оптимален для статичной камеры в помещении: не зависит от угла съёмки
// и формы силуэта, в отличие от HOG.
type Detector struct {
	mu     sync.Mutex
	backSub gocv.BackgroundSubtractorMOG2
	cfg    DetectorConfig
}

func NewDetector(cfg DetectorConfig) *Detector {
	bs := cfg.BackSub
	backSub := gocv.NewBackgroundSubtractorMOG2WithParams(
		bs.History,
		bs.VarThreshold,
		bs.DetectShadows,
	)
	return &Detector{backSub: backSub, cfg: cfg}
}

func (d *Detector) Close() {
	d.backSub.Close()
}

// Reset сбрасывает модель фона — полезно при изменении освещения или перестановке мебели.
func (d *Detector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.backSub.Close()
	bs := d.cfg.BackSub
	d.backSub = gocv.NewBackgroundSubtractorMOG2WithParams(
		bs.History,
		bs.VarThreshold,
		bs.DetectShadows,
	)
}

// Detect обрабатывает кадр: вычитает фон, находит контуры движущихся объектов.
// Первые ~History кадров модель фона ещё строится — детекция постепенно стабилизируется.
func (d *Detector) Detect(img gocv.Mat, withDebug bool) DetectResult {
	d.mu.Lock()
	defer d.mu.Unlock()

	cfg := d.cfg

	// Шаг 1: ресайз до max_width
	scale := 1.0
	working := img
	var resized gocv.Mat
	if cfg.MaxWidth > 0 && img.Cols() > cfg.MaxWidth {
		scale = float64(cfg.MaxWidth) / float64(img.Cols())
		newH := int(float64(img.Rows()) * scale)
		resized = gocv.NewMat()
		gocv.Resize(img, &resized, image.Point{X: cfg.MaxWidth, Y: newH}, 0, 0, gocv.InterpolationLinear)
		working = resized
		defer resized.Close()
	}

	// Шаг 2: вычитание фона → маска переднего плана
	fgMask := gocv.NewMat()
	defer fgMask.Close()
	d.backSub.Apply(working, &fgMask)

	// Шаг 3: морфология — убираем шум и заполняем дыры внутри людей
	bs := cfg.BackSub
	if bs.MorphErodeSize > 0 {
		kernel := gocv.GetStructuringElement(gocv.MorphRect,
			image.Point{X: bs.MorphErodeSize, Y: bs.MorphErodeSize})
		gocv.Erode(fgMask, &fgMask, kernel)
		kernel.Close()
	}
	if bs.MorphDilateSize > 0 {
		kernel := gocv.GetStructuringElement(gocv.MorphRect,
			image.Point{X: bs.MorphDilateSize, Y: bs.MorphDilateSize})
		gocv.Dilate(fgMask, &fgMask, kernel)
		kernel.Close()
	}

	// Шаг 4: поиск контуров
	contours := gocv.FindContours(fgMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	fgPixels := 0
	if withDebug {
		fgPixels = gocv.CountNonZero(fgMask)
	}

	ct := cfg.Contour
	rawContours := make([]image.Rectangle, 0)
	persons := make([]Person, 0)
	finalRects := make([]image.Rectangle, 0)

	for i := 0; i < contours.Size(); i++ {
		contour := contours.At(i)
		area := gocv.ContourArea(contour)

		r := gocv.BoundingRect(contour)
		if withDebug {
			rawContours = append(rawContours, r)
		}

		if area < ct.MinArea || area > ct.MaxArea {
			continue
		}

		// Масштабируем bbox в координаты оригинального кадра
		orig := r
		if scale != 1.0 {
			orig = scaleRect(r, 1.0/scale)
		}

		if orig.Dx() < ct.MinWidth || orig.Dy() < ct.MinHeight {
			continue
		}

		finalRects = append(finalRects, orig)
		persons = append(persons, Person{
			BoundingBox: orig,
			// Нижний центр bbox — точка пола под человеком
			ImagePoint: image.Point{
				X: orig.Min.X + orig.Dx()/2,
				Y: orig.Max.Y,
			},
		})
	}

	result := DetectResult{Persons: persons}
	if withDebug {
		result.Debug = &DebugInfo{
			WorkingSize:      image.Point{X: working.Cols(), Y: working.Rows()},
			WorkingScale:     scale,
			ForegroundPixels: fgPixels,
			RawContours:      rawContours,
			FinalRects:       finalRects,
		}
	}
	return result
}

func scaleRect(r image.Rectangle, s float64) image.Rectangle {
	return image.Rectangle{
		Min: image.Point{X: int(float64(r.Min.X) * s), Y: int(float64(r.Min.Y) * s)},
		Max: image.Point{X: int(float64(r.Max.X) * s), Y: int(float64(r.Max.Y) * s)},
	}
}

// drawDebug рисует bbox-ы детекций на кадре (для визуальной отладки).
func drawDebug(img gocv.Mat, persons []Person) {
	for _, p := range persons {
		gocv.Rectangle(&img, p.BoundingBox, color.RGBA{0, 255, 0, 255}, 2)
		gocv.Circle(&img, p.ImagePoint, 5, color.RGBA{255, 0, 0, 255}, -1)
	}
}
