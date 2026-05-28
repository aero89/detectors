package detector

import (
	"bufio"
	"fmt"
	"image"
	"log/slog"
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

	outNames := unconnectedLayerNames(d.net)
	slog.Debug("dnn: running forward", "layers", outNames)
	outputs := d.net.ForwardLayers(outNames)
	defer func() {
		for i := range outputs {
			outputs[i].Close()
		}
	}()

	var layerDims []map[string]any
	for i, out := range outputs {
		dims := map[string]any{
			"name": outNames[i],
			"rows": out.Rows(),
			"cols": out.Cols(),
			"type": int(out.Type()),
		}
		layerDims = append(layerDims, dims)
		slog.Debug("dnn: output layer", "name", outNames[i], "rows", out.Rows(), "cols", out.Cols(), "type", out.Type())
	}

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
				"layer_dims":      layerDims,
				"conf_threshold":  dn.ConfThreshold,
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
		var b []image.Rectangle
		var s []float32
		if dn.ModelFormat == "yolov8" {
			b, s = parseOutputYOLOv8(output, workW, workH, dn)
		} else {
			b, s = parseOutputYOLOv4(output, workW, workH, dn)
		}
		boxes = append(boxes, b...)
		scores = append(scores, s...)
	}

	slog.Debug("dnn: raw detections", "count", len(boxes), "conf_threshold", dn.ConfThreshold)
	if len(boxes) == 0 {
		slog.Warn("dnn: no detections passed threshold", "conf_threshold", dn.ConfThreshold)
		return nil, nil, nil
	}

	indices := gocv.NMSBoxes(boxes, scores, float32(dn.ConfThreshold), float32(dn.NMSThreshold))
	slog.Debug("dnn: after NMS", "count", len(indices))

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

// parseOutputYOLOv4 разбирает вывод YOLOv3/v4/v5 в формате (N_boxes, 5+classes).
// Колонки: cx cy w h objectness class0 class1 ...
// Координаты нормированы [0,1] относительно входного размера сети.
func parseOutputYOLOv4(output gocv.Mat, workW, workH int, dn config.DNNConfig) ([]image.Rectangle, []float32) {
	var boxes []image.Rectangle
	var scores []float32

	for i := 0; i < output.Rows(); i++ {
		row := output.RowRange(i, i+1)

		if row.Cols() < 6 {
			row.Close()
			continue
		}

		bestClass, bestClassScore := -1, float32(0)
		for c := 5; c < row.Cols(); c++ {
			if s := row.GetFloatAt(0, c); s > bestClassScore {
				bestClassScore = s
				bestClass = c - 5
			}
		}
		if bestClass != dn.PersonClassID {
			row.Close()
			continue
		}

		objectness := row.GetFloatAt(0, 4)
		finalConf := float64(objectness) * float64(bestClassScore)
		row.Close()

		if finalConf < dn.ConfThreshold {
			continue
		}

		cx := float64(output.GetFloatAt(i, 0)) * float64(workW)
		cy := float64(output.GetFloatAt(i, 1)) * float64(workH)
		bw := float64(output.GetFloatAt(i, 2)) * float64(workW)
		bh := float64(output.GetFloatAt(i, 3)) * float64(workH)

		x := int(cx - bw/2)
		y := int(cy - bh/2)
		boxes = append(boxes, image.Rect(x, y, x+int(bw), y+int(bh)))
		scores = append(scores, float32(finalConf))
	}
	return boxes, scores
}

// parseOutputYOLOv8 разбирает вывод YOLOv8 ONNX.
// Сеть отдаёт 3D тензор (1, 4+classes, N_boxes); Reshape превращает его
// в 2D (numAttrs, numBoxes). Objectness отсутствует — confidence = max(class_scores).
// Координаты в пикселях входного изображения (делятся на InputWidth/Height).
func parseOutputYOLOv8(output gocv.Mat, workW, workH int, dn config.DNNConfig) ([]image.Rectangle, []float32) {
	dims := output.Size()
	slog.Debug("dnn: yolov8 raw output", "dims", dims)

	if len(dims) < 2 {
		slog.Warn("dnn: unexpected YOLOv8 output dims", "dims", dims)
		return nil, nil
	}

	// 3D (1, numAttrs, numBoxes) → 2D (numAttrs, numBoxes)
	mat := output.Reshape(1, dims[len(dims)-2])
	defer mat.Close()

	numAttrs := mat.Rows() // 84 для COCO (4 bbox + 80 classes)
	numBoxes := mat.Cols() // 8400 для 640×640

	if numAttrs < 5 || numBoxes == 0 {
		slog.Warn("dnn: unexpected YOLOv8 shape after reshape", "rows", numAttrs, "cols", numBoxes, "dims", dims)
		return nil, nil
	}

	var boxes []image.Rectangle
	var scores []float32

	for boxIdx := 0; boxIdx < numBoxes; boxIdx++ {
		bestClass, bestConf := -1, float32(0)
		for c := 4; c < numAttrs; c++ {
			if s := mat.GetFloatAt(c, boxIdx); s > bestConf {
				bestConf = s
				bestClass = c - 4
			}
		}

		if bestClass != dn.PersonClassID {
			continue
		}
		if float64(bestConf) < dn.ConfThreshold {
			continue
		}

		// Координаты в пикселях входного изображения → масштаб рабочего кадра
		cx := float64(mat.GetFloatAt(0, boxIdx)) / float64(dn.InputWidth) * float64(workW)
		cy := float64(mat.GetFloatAt(1, boxIdx)) / float64(dn.InputHeight) * float64(workH)
		bw := float64(mat.GetFloatAt(2, boxIdx)) / float64(dn.InputWidth) * float64(workW)
		bh := float64(mat.GetFloatAt(3, boxIdx)) / float64(dn.InputHeight) * float64(workH)

		x := int(cx - bw/2)
		y := int(cy - bh/2)
		boxes = append(boxes, image.Rect(x, y, x+int(bw), y+int(bh)))
		scores = append(scores, bestConf)
	}
	return boxes, scores
}

// unconnectedLayerNames возвращает имена выходных слоёв сети.
// gocv v0.35 не имеет GetUnconnectedOutLayersNames, поэтому комбинируем:
// GetUnconnectedOutLayers() → индексы (1-based) + GetLayerNames() → все имена.
func unconnectedLayerNames(net gocv.Net) []string {
	indices := net.GetUnconnectedOutLayers() // []int, 1-based
	allNames := net.GetLayerNames()          // []string, 0-based
	names := make([]string, 0, len(indices))
	for _, idx := range indices {
		if idx > 0 && idx <= len(allNames) {
			names = append(names, allNames[idx-1])
		}
	}
	return names
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
