package detector

import (
	"image"
	"sync"

	"gocv.io/x/gocv"
	"hm-detector/internal/config"
)

// backSubDetector — детектор на основе вычитания фона (MOG2).
// Обнаруживает только движущихся людей, не требует файлов модели.
// Хорош для видеопотока со статичной камерой.
type backSubDetector struct {
	mu      sync.Mutex
	backSub gocv.BackgroundSubtractorMOG2
	cfg     config.DetectorConfig
}

func newBackSubDetector(cfg config.DetectorConfig) *backSubDetector {
	bs := cfg.BackSub
	return &backSubDetector{
		backSub: gocv.NewBackgroundSubtractorMOG2WithParams(
			bs.History, bs.VarThreshold, bs.DetectShadows,
		),
		cfg: cfg,
	}
}

func (d *backSubDetector) Close() {
	d.backSub.Close()
}

func (d *backSubDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.backSub.Close()
	bs := d.cfg.BackSub
	d.backSub = gocv.NewBackgroundSubtractorMOG2WithParams(
		bs.History, bs.VarThreshold, bs.DetectShadows,
	)
}

func (d *backSubDetector) Detect(imgBytes []byte, withDebug bool) (DetectResult, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	img, err := DecodeImage(imgBytes)
	if err != nil {
		return DetectResult{}, err
	}
	defer img.Close()

	scale, working, cleanup := resizeToMaxWidth(img, d.cfg.MaxWidth)
	defer cleanup()

	fgMask := gocv.NewMat()
	defer fgMask.Close()
	d.backSub.Apply(working, &fgMask)

	bs := d.cfg.BackSub
	if bs.MorphErodeSize > 0 {
		k := gocv.GetStructuringElement(gocv.MorphRect,
			image.Point{X: bs.MorphErodeSize, Y: bs.MorphErodeSize})
		gocv.Erode(fgMask, &fgMask, k)
		k.Close()
	}
	if bs.MorphDilateSize > 0 {
		k := gocv.GetStructuringElement(gocv.MorphRect,
			image.Point{X: bs.MorphDilateSize, Y: bs.MorphDilateSize})
		gocv.Dilate(fgMask, &fgMask, k)
		k.Close()
	}

	contours := gocv.FindContours(fgMask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	defer contours.Close()

	ct := d.cfg.Contour
	var rawRects []image.Rectangle
	persons := make([]Person, 0)

	for i := 0; i < contours.Size(); i++ {
		c := contours.At(i)
		area := gocv.ContourArea(c)
		r := gocv.BoundingRect(c)
		if withDebug {
			rawRects = append(rawRects, r)
		}
		if area < ct.MinArea || area > ct.MaxArea {
			continue
		}
		orig := r
		if scale != 1.0 {
			orig = scaleRect(r, 1.0/scale)
		}
		if orig.Dx() < ct.MinWidth || orig.Dy() < ct.MinHeight {
			continue
		}
		persons = append(persons, Person{
			BoundingBox: orig,
			ImagePoint:  footPoint(orig),
		})
	}

	result := DetectResult{Persons: persons}
	if withDebug {
		result.Debug = &DebugInfo{
			WorkingSize:  image.Point{X: working.Cols(), Y: working.Rows()},
			WorkingScale: scale,
			Method:       "backsub",
			Extra: map[string]any{
				"foreground_pixels":    gocv.CountNonZero(fgMask),
				"raw_contours_count":   len(rawRects),
				"raw_contours_working": rawRects,
			},
		}
	}
	return result, nil
}
