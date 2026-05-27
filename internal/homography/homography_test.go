package homography

import (
	"math"
	"testing"
)

func TestNew_TooFewPoints(t *testing.T) {
	_, err := New([]CalibrationPoint{
		{Image: Point2D{0, 0}, Room: Point2D{0, 0}},
		{Image: Point2D{1, 0}, Room: Point2D{1, 0}},
	})
	if err == nil {
		t.Error("expected error for < 4 points")
	}
}

// Прямоугольное помещение 5×4 м, 4 угловые калибровочные точки.
func TestTransform_RectRoom(t *testing.T) {
	pts := []CalibrationPoint{
		{Image: Point2D{120, 680}, Room: Point2D{0, 0}},
		{Image: Point2D{1160, 680}, Room: Point2D{5, 0}},
		{Image: Point2D{1160, 220}, Room: Point2D{5, 4}},
		{Image: Point2D{120, 220}, Room: Point2D{0, 4}},
	}
	hom, err := New(pts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	cases := []struct {
		img  Point2D
		want Point2D
	}{
		{Point2D{120, 680}, Point2D{0, 0}},
		{Point2D{1160, 680}, Point2D{5, 0}},
		{Point2D{1160, 220}, Point2D{5, 4}},
		{Point2D{120, 220}, Point2D{0, 4}},
		// Центр изображения ≈ центр комнаты (только для аффинного случая)
		// при реальной перспективе будет немного отличаться
		{Point2D{640, 450}, Point2D{2.5, 2.0}},
	}
	const tol = 1e-4
	for _, c := range cases {
		got := hom.Transform(c.img)
		if math.Abs(got.X-c.want.X) > tol || math.Abs(got.Y-c.want.Y) > tol {
			t.Errorf("Transform(%v) = {%.5f, %.5f}, want {%.5f, %.5f}",
				c.img, got.X, got.Y, c.want.X, c.want.Y)
		}
	}
}

// Перспективное помещение: точки не образуют прямоугольник на кадре.
func TestTransform_Perspective(t *testing.T) {
	pts := []CalibrationPoint{
		{Image: Point2D{300, 700}, Room: Point2D{0, 0}},
		{Image: Point2D{980, 700}, Room: Point2D{6, 0}},
		{Image: Point2D{1200, 150}, Room: Point2D{6, 8}},
		{Image: Point2D{80, 150}, Room: Point2D{0, 8}},
	}
	hom, err := New(pts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Угловые точки должны отображаться точно
	const tol = 1e-4
	for _, c := range pts {
		got := hom.Transform(c.Image)
		if math.Abs(got.X-c.Room.X) > tol || math.Abs(got.Y-c.Room.Y) > tol {
			t.Errorf("Transform(%v) = {%.5f, %.5f}, want {%.5f, %.5f}",
				c.Image, got.X, got.Y, c.Room.X, c.Room.Y)
		}
	}
}

// Избыточная система (5 точек).
func TestTransform_Overdetermined(t *testing.T) {
	pts := []CalibrationPoint{
		{Image: Point2D{100, 700}, Room: Point2D{0, 0}},
		{Image: Point2D{1180, 700}, Room: Point2D{6, 0}},
		{Image: Point2D{1180, 200}, Room: Point2D{6, 5}},
		{Image: Point2D{100, 200}, Room: Point2D{0, 5}},
		{Image: Point2D{640, 450}, Room: Point2D{3, 2.5}},
	}
	hom, err := New(pts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const tol = 0.05
	for _, c := range pts {
		got := hom.Transform(c.Image)
		if math.Abs(got.X-c.Room.X) > tol || math.Abs(got.Y-c.Room.Y) > tol {
			t.Errorf("Transform(%v) = {%.4f, %.4f}, want {%.4f, %.4f}",
				c.Image, got.X, got.Y, c.Room.X, c.Room.Y)
		}
	}
}
