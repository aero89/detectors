package controller

import (
	"hm-detector/internal/detector"
	"image"
	"log/slog"
	"strings"

	"github.com/gofiber/fiber/v2"
	"gocv.io/x/gocv"
	"hm-detector/internal/homography"
)

// DetectHandler обрабатывает POST /detect.
type DetectHandler struct {
	Detector   *detector.Detector
	Homography *homography.Homography
}

// DetectResponse — ответ ручки /detect.
type DetectResponse struct {
	Persons   []detector.Person   `json:"persons"`
	FrameSize image.Point         `json:"frame_size"`
	Debug     *detector.DebugInfo `json:"debug,omitempty"`
}

// Detect принимает изображение и возвращает список обнаруженных людей.
//
// Принимает изображение двумя способами:
//   - multipart/form-data: поле "image"
//   - любой другой Content-Type: тело запроса целиком
//
// Добавь ?debug=true чтобы получить промежуточные данные каждого шага:
//
//	curl -X POST "http://localhost:8080/detect?debug=true" \
//	     --data-binary @frame.jpg -H "Content-Type: image/jpeg" | jq .debug
func (h *DetectHandler) Detect(c *fiber.Ctx) error {
	withDebug := c.QueryBool("debug", false)

	imgBytes, err := readImageBytes(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "read image: " + err.Error()})
	}

	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "failed to decode image"})
	}
	defer mat.Close()

	result := h.Detector.Detect(mat, withDebug)

	if h.Homography != nil {
		for i := range result.Persons {
			p := result.Persons[i].ImagePoint
			rp := h.Homography.Transform(homography.Point2D{
				X: float64(p.X),
				Y: float64(p.Y),
			})
			result.Persons[i].RoomPoint = &rp
		}
	}

	slog.Info("detect", "persons", len(result.Persons),
		"frame_w", mat.Cols(), "frame_h", mat.Rows(), "debug", withDebug)

	return c.JSON(DetectResponse{
		Persons:   result.Persons,
		FrameSize: image.Point{X: mat.Cols(), Y: mat.Rows()},
		Debug:     result.Debug,
	})
}

func readImageBytes(c *fiber.Ctx) ([]byte, error) {
	ct := c.Get(fiber.HeaderContentType)
	if strings.HasPrefix(ct, fiber.MIMEMultipartForm) {
		file, err := c.FormFile("image")
		if err != nil {
			return nil, err
		}
		f, err := file.Open()
		if err != nil {
			return nil, err
		}
		defer f.Close()
		buf := make([]byte, file.Size)
		if _, err := f.Read(buf); err != nil {
			return nil, err
		}
		return buf, nil
	}
	return c.Body(), nil
}
