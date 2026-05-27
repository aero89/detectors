package main

import (
	"image"
	"log/slog"
	"strings"

	"github.com/aero89/detectors/internal/homography"
	"github.com/gofiber/fiber/v2"
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
// с позициями в пикселях и в координатах помещения.
//
// Принимает изображение двумя способами:
//   - multipart/form-data: поле "image"
//   - любой другой Content-Type: тело запроса целиком
//
// Пример:
//
//	curl -X POST http://localhost:8080/detect \
//	     --data-binary @frame.jpg -H "Content-Type: image/jpeg"
func (h *DetectHandler) Detect(c *fiber.Ctx) error {
	imgBytes, err := readImageBytes(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "read image: " + err.Error()})
	}

	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "failed to decode image"})
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

	return c.JSON(DetectResponse{
		Persons:   persons,
		FrameSize: image.Point{X: mat.Cols(), Y: mat.Rows()},
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
