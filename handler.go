package main

import (
	"encoding/json"
	"fmt"
	"image"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/aero89/detectors/internal/homography"
	"gocv.io/x/gocv"
)

// DetectHandler обрабатывает POST /detect.
type DetectHandler struct {
	detector   *Detector
	homography *homography.Homography
}

// DetectResponse — ответ ручки /detect.
type DetectResponse struct {
	Persons   []Person    `json:"persons"`
	FrameSize image.Point `json:"frame_size"`
}

// Detect принимает изображение и возвращает список обнаруженных людей
// с их позициями как в пикселях, так и в координатах помещения.
//
// Принимает изображение двумя способами:
//   - multipart/form-data: поле "image"
//   - любой другой Content-Type: тело запроса целиком
//
// Пример:
//
//	curl -X POST http://localhost:8080/detect \
//	     --data-binary @frame.jpg -H "Content-Type: image/jpeg"
func (h *DetectHandler) Detect(w http.ResponseWriter, r *http.Request) {
	imgBytes, err := readImageBytes(r)
	if err != nil {
		httpError(w, fmt.Sprintf("read image: %v", err), http.StatusBadRequest)
		return
	}

	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		httpError(w, "failed to decode image", http.StatusBadRequest)
		return
	}
	defer mat.Close()

	persons := h.detector.Detect(mat)

	if h.homography != nil {
		for i := range persons {
			p := persons[i].ImagePoint
			rp := h.homography.Transform(homography.Point2D{
				X: float64(p.X),
				Y: float64(p.Y),
			})
			persons[i].RoomPoint = &rp
		}
	}

	slog.Info("detect", "persons", len(persons),
		"frame_w", mat.Cols(), "frame_h", mat.Rows())

	resp := DetectResponse{
		Persons:   persons,
		FrameSize: image.Point{X: mat.Cols(), Y: mat.Rows()},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func readImageBytes(r *http.Request) ([]byte, error) {
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			return nil, err
		}
		f, _, err := r.FormFile("image")
		if err != nil {
			return nil, fmt.Errorf("field 'image' not found: %w", err)
		}
		defer f.Close()
		return io.ReadAll(f)
	}
	defer r.Body.Close()
	return io.ReadAll(r.Body)
}

func httpError(w http.ResponseWriter, msg string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}
