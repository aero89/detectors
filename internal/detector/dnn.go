package detector

import (
	"bufio"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
	"hm-detector/internal/config"
)

// dnnDetector — детектор на основе нейросети (YOLO / MobileNet через OpenCV DNN).
// Работает на одном кадре, не требует видеопотока. Угол камеры не важен.
//
// Поддерживаемые форматы моделей:
//   - Darknet YOLO: .weights + .cfg  (YOLOv3, YOLOv4, tiny-варианты)
//   - ONNX:         .onnx            (YOLOv5, YOLOv8, MobileNet и др.)
type dnnDetector struct {
	net        gocv.Net
	classNames []string
	cfg        config.DetectorConfig
}

func newDNNDetector(cfg config.DetectorConfig) (*dnnDetector, error) {
	dn := cfg.DNN
	if dn.Model == "" {
		return nil, fmt.Errorf("dnn.model path is required")
	}

	var net gocv.Net
	ext := strings.ToLower(filepath.Ext(dn.Model))
	switch ext {
	case ".weights":
		if dn.Config == "" {
			return nil, fmt.Errorf("dnn.config (.cfg file) is required for Darknet models")
		}
		// ReadNet(model, config) — универсальный загрузчик OpenCV DNN,
		// определяет формат по расширению файла.
		net = gocv.ReadNet(dn.Model, dn.Config)
	case ".onnx":
		net = gocv.ReadNet(dn.Model, "")
	default:
		return nil, fmt.Errorf("unsupported model format %q (expected .weights or .onnx)", ext)
	}

	if net.Empty() {
		return nil, fmt.Errorf("failed to load model from %s", dn.Model)
	}

	net.SetPreferableBackend(gocv.NetBackendType(dn.Backend))
	net.SetPreferableTarget(gocv.NetTargetType(dn.Target))

	classNames, err := loadClassNames(dn.Classes)
	if err != nil {
		return nil, fmt.Errorf("load class names: %w", err)
	}

	return &dnnDetector{net: net, classNames: classNames, cfg: cfg}, nil
}

func (d *dnnDetector) Close() { d.net.Close() }
func (d *dnnDetector) Reset() {}

func (d *dnnDetector) Detect(imgBytes []byte, withDebug bool) (DetectResult, error) {
	img, err := DecodeImage(imgBytes)
	if err != nil {
		return DetectResult{}, err
	}
	defer img.Close()

	dn := d.cfg.DNN
	scale, working, cleanup := resizeToMaxWidth(img, d.cfg.MaxWidth)
	defer cleanup()

	blob := gocv.BlobFromImage(
		working,
		1.0/255.0,
		image.Pt(dn.InputWidth, dn.InputHeight),
		gocv.NewScalar(0, 0, 0, 0),
		true,
		false,
	)
	defer blob.Close()

	d.net.SetInput(blob, "")

	outNames := d.net.GetUnconnectedOutLayersNames()
	outputs := d.net.ForwardLayers(outNames)
	defer func() {
		for i := range outputs {
			outputs[i].Close()
		}
	}()

	persons, rawBoxes, rawScores := d.parseDetections(outputs, working.Cols(), working.Rows(), scale)

	result := DetectResult{Persons: persons}
	if withDebug {
		result.Debug = &DebugInfo{
			WorkingSize:  image.Point{X: working.Cols(), Y: working.Rows()},
			WorkingScale: scale,
			Method:       "dnn",
			Extra: map[string]any{
				"raw_boxes_count": len(rawBoxes),
				"raw_boxes":       rawBoxes,
				"raw_scores":      rawScores,
			},
		}
	}
	return result, nil
}

func (d *dnnDetector) parseDetections(outputs []gocv.Mat, workW, workH int, scale float64) ([]Person, []image.Rectangle, []float32) {
	dn := d.cfg.DNN
	ct := d.cfg.Contour

	var boxes []image.Rectangle
	var scores []float32

	for _, output := range outputs {
		for i := 0; i < output.Rows(); i++ {
			row := output.RowRange(i, i+1)
			if row.Cols() < 5 {
				row.Close()
				continue
			}

			confidence := float64(row.GetFloatAt(0, 4))
			if confidence < dn.ConfThreshold {
				row.Close()
				continue
			}

			bestClass, bestScore := 0, float32(0)
			for c := 5; c < row.Cols(); c++ {
				if s := row.GetFloatAt(0, c); s > bestScore {
					bestScore = s
					bestClass = c - 5
				}
			}
			row.Close()

			if bestClass != dn.PersonClassID {
				continue
			}
			if float64(bestScore)*confidence < dn.ConfThreshold {
				continue
			}

			cx := float64(output.GetFloatAt(i, 0)) * float64(workW)
			cy := float64(output.GetFloatAt(i, 1)) * float64(workH)
			w := float64(output.GetFloatAt(i, 2)) * float64(workW)
			h := float64(output.GetFloatAt(i, 3)) * float64(workH)

			x := int(cx - w/2)
			y := int(cy - h/2)
			boxes = append(boxes, image.Rect(x, y, x+int(w), y+int(h)))
			scores = append(scores, bestScore*float32(confidence))
		}
	}

	if len(boxes) == 0 {
		return nil, nil, nil
	}

	indices := gocv.NMSBoxes(boxes, scores, float32(dn.ConfThreshold), float32(dn.NMSThreshold))

	persons := make([]Person, 0, len(indices))
	for _, idx := range indices {
		r := boxes[idx]
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
	return persons, boxes, scores
}

func loadClassNames(path string) ([]string, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var names []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if name := strings.TrimSpace(sc.Text()); name != "" {
			names = append(names, name)
		}
	}
	return names, sc.Err()
}
