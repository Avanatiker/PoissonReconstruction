/// 3D geometry primitives: Point3, Vec3, BoundingBox, Matrix4.
///
/// Lightweight implementation matching the C++ Geometry.h types
/// without external dependencies.

use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Vec3 = Vec3 { x: 1.0, y: 1.0, z: 1.0 };

    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn dot(&self, other: &Vec3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn cross(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    #[inline]
    pub fn square_norm(&self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(&self) -> f64 {
        self.square_norm().sqrt()
    }

    #[inline]
    pub fn normalize(&self) -> Vec3 {
        let n = self.norm();
        if n > 0.0 {
            *self / n
        } else {
            Vec3::ZERO
        }
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of bounds: {}", i),
        }
    }
}

// Vec3 + Vec3
impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Vec3) { *self = *self + rhs; }
}
impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec3) { *self = *self - rhs; }
}

// Vec3 * scalar
impl Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 { v * self }
}
impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, s: f64) { *self = *self * s; }
}
impl Div<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}
impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, s: f64) { *self = *self / s; }
}
impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

// Point3: a position in 3D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const ORIGIN: Point3 = Point3 { x: 0.0, y: 0.0, z: 0.0 };

    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self { Point3 { x, y, z } }

    #[inline]
    pub fn from_slice(coords: &[f64]) -> Self { Point3 { x: coords[0], y: coords[1], z: coords[2] } }
}

impl Add for Point3 {
    type Output = Point3;
    #[inline]
    fn add(self, rhs: Point3) -> Point3 { Point3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}

impl Mul<f64> for Point3 {
    type Output = Point3;
    #[inline]
    fn mul(self, s: f64) -> Point3 { Point3::new(self.x * s, self.y * s, self.z * s) }
}

impl Default for Point3 {
    fn default() -> Self { Point3::ORIGIN }
}

impl Index<usize> for Point3 {
    type Output = f64;
    #[inline]
    fn index(&self, i: usize) -> &f64 {
        match i {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Point3 index out of bounds: {}", i),
        }
    }
}

impl IndexMut<usize> for Point3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        match i {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Point3 index out of bounds: {}", i),
        }
    }
}

// Point3 + Vec3 = Point3
impl Add<Vec3> for Point3 {
    type Output = Point3;
    #[inline]
    fn add(self, rhs: Vec3) -> Point3 {
        Point3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign<Vec3> for Point3 {
    #[inline]
    fn add_assign(&mut self, rhs: Vec3) { *self = *self + rhs; }
}
impl Sub<Vec3> for Point3 {
    type Output = Point3;
    #[inline]
    fn sub(self, rhs: Vec3) -> Point3 {
        Point3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl SubAssign<Vec3> for Point3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Vec3) { *self = *self - rhs; }
}
// Point3 - Point3 = Vec3
impl Sub for Point3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, rhs: Point3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

// 4x4 homogeneous transformation matrix (column-major)
#[derive(Debug, Clone, Copy)]
pub struct Mat4 {
    /// Column-major: m[col][row]
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = [[0.0f64; 4]; 4];
        for i in 0..4 {
            m[i][i] = 1.0;
        }
        Mat4 { m }
    }

    pub fn zeros() -> Self {
        Mat4 { m: [[0.0; 4]; 4] }
    }

    /// Transform a 3D point (x, y, z, 1)
    pub fn transform_point(&self, p: &Point3) -> Point3 {
        let x = self.m[0][0] * p.x + self.m[1][0] * p.y + self.m[2][0] * p.z + self.m[3][0];
        let y = self.m[0][1] * p.x + self.m[1][1] * p.y + self.m[2][1] * p.z + self.m[3][1];
        let z = self.m[0][2] * p.x + self.m[1][2] * p.y + self.m[2][2] * p.z + self.m[3][2];
        Point3::new(x, y, z)
    }

    /// Transform a 3D vector (ignoring translation)
    pub fn transform_vector(&self, v: &Vec3) -> Vec3 {
        let x = self.m[0][0] * v.x + self.m[1][0] * v.y + self.m[2][0] * v.z;
        let y = self.m[0][1] * v.x + self.m[1][1] * v.y + self.m[2][1] * v.z;
        let z = self.m[0][2] * v.x + self.m[1][2] * v.y + self.m[2][2] * v.z;
        Vec3::new(x, y, z)
    }

    /// Compute the inverse of a 4x4 affine matrix.
    ///
    /// For affine: [R | t; 0 0 0 1], inverse = [R^{-1} | -R^{-1}*t; 0 0 0 1]
    pub fn inverse(&self) -> Mat4 {
        // Extract 3x3 linear part and invert
        let a = self.m[0][0]; let b = self.m[1][0]; let c = self.m[2][0];
        let d = self.m[0][1]; let e = self.m[1][1]; let f = self.m[2][1];
        let g = self.m[0][2]; let h = self.m[1][2]; let i = self.m[2][2];

        // Compute inverse of 3x3 using cofactor method
        let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

        if det.abs() < 1e-30 {
            return Mat4::identity();
        }

        let inv_det = 1.0 / det;

        let a_inv = (e * i - f * h) * inv_det;
        let b_inv = (c * h - b * i) * inv_det;
        let c_inv = (b * f - c * e) * inv_det;
        let d_inv = (f * g - d * i) * inv_det;
        let e_inv = (a * i - c * g) * inv_det;
        let f_inv = (c * d - a * f) * inv_det;
        let g_inv = (d * h - e * g) * inv_det;
        let h_inv = (b * g - a * h) * inv_det;
        let i_inv = (a * e - b * d) * inv_det;

        let tx = self.m[3][0];
        let ty = self.m[3][1];
        let tz = self.m[3][2];

        // -R^{-1} * t
        let inv_tx = -(a_inv * tx + b_inv * ty + c_inv * tz);
        let inv_ty = -(d_inv * tx + e_inv * ty + f_inv * tz);
        let inv_tz = -(g_inv * tx + h_inv * ty + i_inv * tz);

        Mat4 {
            m: [
                [a_inv, d_inv, g_inv, 0.0],
                [b_inv, e_inv, h_inv, 0.0],
                [c_inv, f_inv, i_inv, 0.0],
                [inv_tx, inv_ty, inv_tz, 1.0],
            ],
        }
    }

    /// Get the 3x3 linear part as an array.
    pub fn linear_3x3(&self) -> [[f64; 3]; 3] {
        [
            [self.m[0][0], self.m[1][0], self.m[2][0]],
            [self.m[0][1], self.m[1][1], self.m[2][1]],
            [self.m[0][2], self.m[1][2], self.m[2][2]],
        ]
    }

    /// Multiply self (on left) with another matrix: self * other
    pub fn compose(&self, other: &Mat4) -> Mat4 {
        let mut result = Mat4::zeros();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result.m[i][j] += self.m[k][j] * other.m[i][k];
                }
            }
        }
        result
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub min: Point3,
    pub max: Point3,
}

impl BBox {
    pub fn empty() -> Self {
        BBox {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    pub fn extend(&mut self, p: &Point3) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn max_extent(&self) -> f64 {
        let s = self.size();
        s.x.max(s.y).max(s.z)
    }
}

/// Compute a 4x4 transform that maps a bounding box (scaled by `scale`) into the unit cube [0,1]^3.
pub fn unit_cube_transform(bbox: &BBox, scale: f64) -> Mat4 {
    let center = bbox.center();
    let max_dim = bbox.max_extent();
    let s = if max_dim > 0.0 { scale / max_dim } else { 1.0 };

    // Rows of the matrix (column-major storage):
    // Column 0: scale_x, 0, 0, 0
    // Column 1: 0, scale_y, 0, 0
    // Column 2: 0, 0, scale_z, 0
    // Column 3: translate_x, translate_y, translate_z, 1
    Mat4 {
        m: [
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, s, 0.0],
            [-s * center.x + 0.5, -s * center.y + 0.5, -s * center.z + 0.5, 1.0],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_ops() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert_eq!(c, Vec3::new(5.0, 7.0, 9.0));

        let d = b - a;
        assert_eq!(d, Vec3::new(3.0, 3.0, 3.0));

        let dot = a.dot(&b);
        assert!((dot - 32.0).abs() < 1e-10);

        let cross = a.cross(&b);
        assert_eq!(cross, Vec3::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_mat4_inverse_identity() {
        let m = Mat4::identity();
        let inv = m.inverse();
        // Check that m * inv is identity
        let p = Point3::new(1.0, 2.0, 3.0);
        let q = m.transform_point(&p);
        let r = inv.transform_point(&q);
        assert!((r.x - p.x).abs() < 1e-10);
        assert!((r.y - p.y).abs() < 1e-10);
        assert!((r.z - p.z).abs() < 1e-10);
    }

    #[test]
    fn test_bbox() {
        let mut bbox = BBox::empty();
        bbox.extend(&Point3::new(-1.0, -2.0, -3.0));
        bbox.extend(&Point3::new(1.0, 2.0, 3.0));
        assert_eq!(bbox.center(), Point3::new(0.0, 0.0, 0.0));
        assert_eq!(bbox.size(), Vec3::new(2.0, 4.0, 6.0));
        assert!((bbox.max_extent() - 6.0).abs() < 1e-10);
    }
}
