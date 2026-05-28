package main

import (
	"fmt"
	"image"

	"gocv.io/x/gocv"
)

func decodeImage(imgBytes []byte) (gocv.Mat, error) {
	mat, err := gocv.IMDecode(imgBytes, gocv.IMReadColor)
	if err != nil || mat.Empty() {
		return gocv.NewMat(), fmt.Errorf("failed to decode image")
	}
	return mat, nil
}

// resizeToMaxWidth масштабирует изображение если оно шире maxWidth.
// Возвращает scale (working/original), рабочий Mat и функцию освобождения.
func resizeToMaxWidth(img gocv.Mat, maxWidth int) (scale float64, working gocv.Mat, cleanup func()) {
	if maxWidth <= 0 || img.Cols() <= maxWidth {
		return 1.0, img, func() {}
	}
	scale = float64(maxWidth) / float64(img.Cols())
	newH := int(float64(img.Rows()) * scale)
	resized := gocv.NewMat()
	gocv.Resize(img, &resized, image.Point{X: maxWidth, Y: newH}, 0, 0, gocv.InterpolationLinear)
	return scale, resized, func() { resized.Close() }
}
