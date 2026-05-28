package main

import (
	"bufio"
	"fmt"
	"image"
	"os"
	"path/filepath"
	"strings"

	"gocv.io/x/gocv"
)

// DNNDetector — детектор на основе нейросети (YOLO / MobileNet через OpenCV DNN).
// Работает на одном кадре, не требует видеопотока. Угол камеры не важен.
//
// Поддерживаемые форматы моделей:
//   - Darknet YOLO:   .weights + .cfg  (YOLOv3, YOLOv4, tiny-варианты)
//   - ONNX:           .onnx            (YOLOv5, YOLOv8, MobileNet и др.)
type DNNDetector struct {
	net        gocv.Net
	classNames []string
	cfg        DetectorConfig
}

func NewDNNDetector(cfg DetectorConfig) (*DNNDetector, error) {
	dn := cfg.DNN
	if dn.Model == "" {
		return nil, fmt.Errorf("dnn.model path is required")
	}

	var net gocv.Net
	ext := strings.ToLower(filepath.Ext(dn.Model))
	switch ext {
	case ".weights":
		// Darknet YOLO
		if dn.Config == "" {
			return nil, fmt.Errorf("dnn.config (.cfg file) is required for Darknet models")
		}
		net = gocv.ReadNetFromDarknet(dn.Config, dn.Model)
	case ".onnx":
		net = gocv.ReadNetFromONNX(dn.Model)
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

	return &DNNDetector{net: net, classNames: classNames, cfg: cfg}, nil
}

func (d *DNNDetector) Close() { d.net.Close() }
func (d *DNNDetector) Reset() {} // DNN не имеет состояния кадра

func (d *DNNDetector) Detect(imgBytes []byte, withDebug bool) (DetectResult, error) {
	img, err := decodeImage(imgBytes)
	if err != nil {
		return DetectResult{}, err
	}
	defer img.Close()

	cfg := d.cfg
	dn := cfg.DNN

	scale, working, cleanup := resizeToMaxWidth(img, cfg.MaxWidth)
	defer cleanup()

	// Формируем входной blob
	blob := gocv.BlobFromImage(
		working,
		1.0/255.0,
		image.Pt(dn.InputWidth, dn.InputHeight),
		gocv.NewScalar(0, 0, 0, 0),
		true,  // swapRB
		false, // crop
	)
	defer blob.Close()

	d.net.SetInput(blob, "")

	// Прямой проход
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

// parseDetections разбирает выходы YOLO-сети и применяет NMS.
func (d *DNNDetector) parseDetections(outputs []gocv.Mat, workW, workH int, invScale float64) ([]Person, []image.Rectangle, []float32) {
	dn := d.cfg.DNN
	ct := d.cfg.Contour

	var boxes []image.Rectangle
	var scores []float32
	var classIDs []int

	for _, output := range outputs {
		for i := 0; i < output.Rows(); i++ {
			row := output.RowRange(i, i+1)
			// YOLO строка: [cx, cy, w, h, objectness, class0, class1, ...]
			if row.Cols() < 5 {
				row.Close()
				continue
			}

			confidence := float64(row.GetFloatAt(0, 4))
			if confidence < dn.ConfThreshold {
				row.Close()
				continue
			}

			// Найдём класс с максимальным score
			bestClass, bestScore := 0, float32(0)
			for c := 5; c < row.Cols(); c++ {
				s := row.GetFloatAt(0, c)
				if s > bestScore {
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

			// cx, cy, w, h — относительные координаты (0–1)
			cx := float64(output.GetFloatAt(i, 0)) * float64(workW)
			cy := float64(output.GetFloatAt(i, 1)) * float64(workH)
			w := float64(output.GetFloatAt(i, 2)) * float64(workW)
			h := float64(output.GetFloatAt(i, 3)) * float64(workH)

			x := int(cx - w/2)
			y := int(cy - h/2)
			boxes = append(boxes, image.Rect(x, y, x+int(w), y+int(h)))
			scores = append(scores, bestScore*float32(confidence))
			classIDs = append(classIDs, bestClass)
		}
	}

	if len(boxes) == 0 {
		return nil, nil, nil
	}

	// NMS
	indices := gocv.NMSBoxes(boxes, scores, float32(dn.ConfThreshold), float32(dn.NMSThreshold))

	persons := make([]Person, 0, len(indices))
	rawBoxes := make([]image.Rectangle, 0, len(boxes))
	rawBoxes = append(rawBoxes, boxes...)

	for _, idx := range indices {
		r := boxes[idx]
		orig := r
		if invScale != 1.0 {
			orig = scaleRect(r, 1.0/invScale)
		}
		if orig.Dx() < ct.MinWidth || orig.Dy() < ct.MinHeight {
			continue
		}
		persons = append(persons, Person{
			BoundingBox: orig,
			ImagePoint:  footPoint(orig),
		})
	}
	return persons, rawBoxes, scores
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
