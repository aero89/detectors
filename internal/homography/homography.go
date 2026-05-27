// Package homography вычисляет проективное преобразование (гомографию)
// для перевода пиксельных координат изображения в координаты помещения.
package homography

import (
	"errors"
	"fmt"
	"math"
)

// Point2D — двумерная точка с вещественными координатами.
type Point2D struct {
	X, Y float64
}

// CalibrationPoint задаёт соответствие между точкой на изображении и
// точкой в системе координат помещения.
type CalibrationPoint struct {
	Image Point2D
	Room  Point2D
}

// Homography хранит матрицу 3×3 для проективного преобразования.
// Вычисляется методом DLT по N >= 4 парам точек.
type Homography struct {
	h [9]float64
}

// New вычисляет матрицу гомографии по калибровочным точкам.
// Нужно минимум 4 пары. Чем больше точек — тем точнее результат.
func New(pts []CalibrationPoint) (*Homography, error) {
	if len(pts) < 4 {
		return nil, fmt.Errorf("homography requires at least 4 calibration points, got %d", len(pts))
	}

	srcRaw := make([]Point2D, len(pts))
	dstRaw := make([]Point2D, len(pts))
	for i, p := range pts {
		srcRaw[i] = p.Image
		dstRaw[i] = p.Room
	}

	srcNorm, Ts := normalizePoints(srcRaw)
	dstNorm, Td := normalizePoints(dstRaw)

	h, err := dltHomography(srcNorm, dstNorm)
	if err != nil {
		return nil, err
	}

	// Денормализация: H_orig = Td^-1 * H_norm * Ts
	hFull := mat3Mul(mat3Mul(mat3Inv(Td), h), Ts)

	if math.Abs(hFull[8]) > 1e-12 {
		for i := range hFull {
			hFull[i] /= hFull[8]
		}
	}

	return &Homography{h: hFull}, nil
}

// Transform переводит точку из пиксельных координат в координаты помещения.
func (hg *Homography) Transform(p Point2D) Point2D {
	h := hg.h
	wx := h[0]*p.X + h[1]*p.Y + h[2]
	wy := h[3]*p.X + h[4]*p.Y + h[5]
	w := h[6]*p.X + h[7]*p.Y + h[8]
	return Point2D{X: wx / w, Y: wy / w}
}

// normalizePoints центрирует и масштабирует точки для численной устойчивости DLT.
// Возвращает нормализованные точки и матрицу трансформации T (3×3).
func normalizePoints(pts []Point2D) ([]Point2D, [9]float64) {
	n := float64(len(pts))
	var cx, cy float64
	for _, p := range pts {
		cx += p.X
		cy += p.Y
	}
	cx /= n
	cy /= n

	var meanDist float64
	for _, p := range pts {
		dx, dy := p.X-cx, p.Y-cy
		meanDist += math.Sqrt(dx*dx + dy*dy)
	}
	meanDist /= n
	if meanDist < 1e-10 {
		meanDist = 1
	}
	scale := math.Sqrt2 / meanDist

	norm := make([]Point2D, len(pts))
	for i, p := range pts {
		norm[i] = Point2D{X: scale * (p.X - cx), Y: scale * (p.Y - cy)}
	}
	T := [9]float64{
		scale, 0, -scale * cx,
		0, scale, -scale * cy,
		0, 0, 1,
	}
	return norm, T
}

// dltHomography вычисляет гомографию по нормализованным точкам.
// Строит систему Ax = b (h[8]=1) и решает нормальные уравнения 8×8.
func dltHomography(src, dst []Point2D) ([9]float64, error) {
	rows := 2 * len(src)
	A := make([]float64, rows*8)
	b := make([]float64, rows)

	for i := range src {
		x, y := src[i].X, src[i].Y
		u, v := dst[i].X, dst[i].Y

		r0 := 2 * i
		A[r0*8+0] = -x; A[r0*8+1] = -y; A[r0*8+2] = -1
		A[r0*8+6] = u * x; A[r0*8+7] = u * y
		b[r0] = -u

		r1 := r0 + 1
		A[r1*8+3] = -x; A[r1*8+4] = -y; A[r1*8+5] = -1
		A[r1*8+6] = v * x; A[r1*8+7] = v * y
		b[r1] = -v
	}

	// Нормальные уравнения: (A^T A) x = A^T b
	var AtA [64]float64
	var Atb [8]float64
	for i := 0; i < rows; i++ {
		for j := 0; j < 8; j++ {
			for k := 0; k < 8; k++ {
				AtA[j*8+k] += A[i*8+j] * A[i*8+k]
			}
			Atb[j] += A[i*8+j] * b[i]
		}
	}

	x, err := gaussElim8(AtA[:], Atb[:])
	if err != nil {
		return [9]float64{}, fmt.Errorf("singular homography system: %w", err)
	}
	var h [9]float64
	copy(h[:8], x[:])
	h[8] = 1.0
	return h, nil
}

// gaussElim8 решает систему 8×8 методом Гаусса с частичным выбором ведущего элемента.
func gaussElim8(A, b []float64) ([8]float64, error) {
	const n = 8
	aug := make([]float64, n*(n+1))
	for i := 0; i < n; i++ {
		copy(aug[i*(n+1):i*(n+1)+n], A[i*n:i*n+n])
		aug[i*(n+1)+n] = b[i]
	}

	for col := 0; col < n; col++ {
		pivot, maxVal := col, math.Abs(aug[col*(n+1)+col])
		for row := col + 1; row < n; row++ {
			if v := math.Abs(aug[row*(n+1)+col]); v > maxVal {
				maxVal, pivot = v, row
			}
		}
		if maxVal < 1e-12 {
			return [8]float64{}, errors.New("matrix is singular or near-singular")
		}
		if pivot != col {
			for k := 0; k <= n; k++ {
				aug[col*(n+1)+k], aug[pivot*(n+1)+k] = aug[pivot*(n+1)+k], aug[col*(n+1)+k]
			}
		}
		for row := col + 1; row < n; row++ {
			f := aug[row*(n+1)+col] / aug[col*(n+1)+col]
			for k := col; k <= n; k++ {
				aug[row*(n+1)+k] -= f * aug[col*(n+1)+k]
			}
		}
	}

	var x [8]float64
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i*(n+1)+n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i*(n+1)+j] * x[j]
		}
		x[i] /= aug[i*(n+1)+i]
	}
	return x, nil
}

func mat3Mul(a, b [9]float64) [9]float64 {
	var c [9]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				c[i*3+j] += a[i*3+k] * b[k*3+j]
			}
		}
	}
	return c
}

func mat3Inv(m [9]float64) [9]float64 {
	det := m[0]*(m[4]*m[8]-m[5]*m[7]) -
		m[1]*(m[3]*m[8]-m[5]*m[6]) +
		m[2]*(m[3]*m[7]-m[4]*m[6])
	return [9]float64{
		(m[4]*m[8] - m[5]*m[7]) / det, (m[2]*m[7] - m[1]*m[8]) / det, (m[1]*m[5] - m[2]*m[4]) / det,
		(m[5]*m[6] - m[3]*m[8]) / det, (m[0]*m[8] - m[2]*m[6]) / det, (m[2]*m[3] - m[0]*m[5]) / det,
		(m[3]*m[7] - m[4]*m[6]) / det, (m[1]*m[6] - m[0]*m[7]) / det, (m[0]*m[4] - m[1]*m[3]) / det,
	}
}
