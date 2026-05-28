package main

import (
	"image"
	"io"
	"log/slog"
	"strings"

	"github.com/aero89/detectors/internal/homography"
	"github.com/gofiber/fiber/v2"
)

type DetectHandler struct {
	detector   PersonDetector
	homography *homography.Homography
}

type DetectResponse struct {
	Persons   []Person    `json:"persons"`
	FrameSize image.Point `json:"frame_size"`
	Debug     *DebugInfo  `json:"debug,omitempty"`
}

// Detect принимает изображение и возвращает список людей.
//
// Способы передачи изображения:
//   - multipart/form-data: поле "image"
//   - любой другой Content-Type: тело запроса целиком (raw bytes)
//
// ?debug=true — добавить промежуточные данные детектора в ответ.
func (h *DetectHandler) Detect(c *fiber.Ctx) error {
	withDebug := c.QueryBool("debug", false)

	imgBytes, err := readImageBytes(c)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "read image: " + err.Error()})
	}

	result, err := h.detector.Detect(imgBytes, withDebug)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}

	if h.homography != nil {
		for i := range result.Persons {
			p := result.Persons[i].ImagePoint
			rp := h.homography.Transform(homography.Point2D{X: float64(p.X), Y: float64(p.Y)})
			result.Persons[i].RoomPoint = &rp
		}
	}

	// Размер кадра из debug или декодируем заново
	frameSize := frameSizeFromDebug(result.Debug, imgBytes)

	slog.Info("detect", "persons", len(result.Persons))

	return c.JSON(DetectResponse{
		Persons:   result.Persons,
		FrameSize: frameSize,
		Debug:     result.Debug,
	})
}

func frameSizeFromDebug(dbg *DebugInfo, imgBytes []byte) image.Point {
	if dbg != nil && dbg.WorkingScale > 0 && dbg.WorkingScale != 1.0 {
		return image.Point{
			X: int(float64(dbg.WorkingSize.X) / dbg.WorkingScale),
			Y: int(float64(dbg.WorkingSize.Y) / dbg.WorkingScale),
		}
	}
	if dbg != nil {
		return dbg.WorkingSize
	}
	// Без debug: быстрый декод только заголовка
	if mat, err := decodeImage(imgBytes); err == nil {
		sz := image.Point{X: mat.Cols(), Y: mat.Rows()}
		mat.Close()
		return sz
	}
	return image.Point{}
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
		_, err = io.ReadFull(f, buf)
		return buf, err
	}
	return c.Body(), nil
}
