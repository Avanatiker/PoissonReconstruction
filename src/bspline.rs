/// B-spline basis functions for the FEM system.
///
/// Implements evaluation of B-spline basis functions of degree 1 and 2
/// (the default FEM degrees used by PoissonRecon), plus dot products
/// for system matrix assembly, and up-sampling/prolongation stencils
/// for multigrid.

/// Boundary condition type for the FEM basis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    Free = 0,
    Dirichlet = 1,
    Neumann = 2,
}

impl BoundaryType {
    pub const COUNT: usize = 3;

    pub fn has_partition_of_unity(&self) -> bool {
        !matches!(self, BoundaryType::Dirichlet)
    }
}

/// A polynomial of degree `degree` stored as a vector of coefficients.
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f64>) -> Self {
        Polynomial { coeffs }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.is_empty() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Evaluate at t using Horner's method.
    pub fn eval(&self, t: f64) -> f64 {
        let mut v = 0.0;
        for i in (0..self.coeffs.len()).rev() {
            v = v * t + self.coeffs[i];
        }
        v
    }

    /// Compute the d-th derivative polynomial.
    pub fn derivative(&self, d: usize) -> Polynomial {
        if d >= self.coeffs.len() {
            return Polynomial::new(vec![0.0]);
        }
        let new_degree = self.coeffs.len() - 1 - d;
        let mut new_coeffs = vec![0.0; new_degree + 1];
        // N_D^{(d)}(t): multiply each coefficient by falling factorial
        for i in 0..=new_degree {
            let mut factor = 1.0f64;
            for k in 0..d {
                factor *= (i + d - k) as f64;
            }
            new_coeffs[i] = self.coeffs[i + d] * factor;
        }
        Polynomial::new(new_coeffs)
    }
}

/// Get the B-spline component polynomials for a given degree.
/// Returns `degree+1` polynomials, one per cell in the support.
/// These are the De Boor B-spline polynomials.
pub fn bspline_components(degree: usize) -> Vec<Polynomial> {
    match degree {
        0 => vec![Polynomial::new(vec![1.0])],
        1 => vec![
            Polynomial::new(vec![1.0, -1.0]), // B_0(t) = 1 - t
            Polynomial::new(vec![0.0, 1.0]),  // B_1(t) = t
        ],
        2 => vec![
            Polynomial::new(vec![0.5, -1.0, 0.5]), // B_0(t) = (1-t)^2/2
            Polynomial::new(vec![0.5, 1.0, -1.0]), // B_1(t) = (1+2t-2t^2)/2
            Polynomial::new(vec![0.0, 0.0, 0.5]),  // B_2(t) = t^2/2
        ],
        _ => {
            // Generic recurrence for higher degrees
            let lower = bspline_components(degree - 1);
            let n = degree + 1;
            let mut result = Vec::with_capacity(n);
            result.push(Polynomial::new(vec![0.0; degree + 1]));
            for _ in 1..n {
                result.push(Polynomial::new(vec![0.0; degree + 1]));
            }

            for i in 0..n {
                if i < degree {
                    let lower_i = &lower[i];
                    let _scale = 1.0 / degree as f64;
                    let integral = lower_i.integral();
                    // B_i^D(t) contribution from B_i^{D-1}
                    for k in 0..integral.coeffs.len() {
                        if k < result[i].coeffs.len() {
                            result[i].coeffs[k] -= integral.coeffs[k];
                        }
                    }
                    // Add integral(1) as constant term
                    result[i].coeffs[0] += integral.eval(1.0);
                }
                if i > 0 {
                    let lower_im1 = &lower[i - 1];
                    let integral = lower_im1.integral();
                    for k in 0..integral.coeffs.len() {
                        if k < result[i].coeffs.len() {
                            result[i].coeffs[k] += integral.coeffs[k];
                        }
                    }
                }
            }
            result
        }
    }
}

impl Polynomial {
    /// Compute the integral polynomial (indefinite integral from 0 to t).
    pub fn integral(&self) -> Polynomial {
        let n = self.coeffs.len();
        let mut new_coeffs = vec![0.0; n + 1];
        // Integral of c_k * t^k is c_k * t^{k+1} / (k+1)
        for k in 0..n {
            new_coeffs[k + 1] = self.coeffs[k] / (k + 1) as f64;
        }
        Polynomial::new(new_coeffs)
    }
}

/// Support sizes for B-spline basis functions.
#[derive(Debug, Clone)]
pub struct BSplineSupport {
    pub support_start: isize,
    pub support_end: isize,
    pub support_size: usize,
    pub child_support_start: isize,
    pub child_support_end: isize,
    pub up_sample_start: isize,
    pub up_sample_end: isize,
    pub up_sample_size: usize,
}

impl BSplineSupport {
    pub fn new(degree: usize) -> Self {
        if degree == 0 {
            return BSplineSupport {
                support_start: 0,
                support_end: 0,
                support_size: 1,
                child_support_start: 0,
                child_support_end: 0,
                up_sample_start: 0,
                up_sample_end: 0,
                up_sample_size: 1,
            };
        }

        let primal = (degree & 1) == 1;
        let half = (degree + 1) / 2;

        let (support_start, support_end) = if primal {
            let s = half as isize;
            (-s + 1, s)
        } else {
            let s = half as isize;
            (-s, s)
        };
        let support_size = (support_end - support_start + 1) as usize;

        let child_support_start = support_start * 2;
        let child_support_end = support_end * 2 + 1;

        let up_sample_start = support_start * 2;
        let up_sample_end = support_end * 2 + 1;
        let up_sample_size = (up_sample_end - up_sample_start + 1) as usize;

        BSplineSupport {
            support_start,
            support_end,
            support_size,
            child_support_start,
            child_support_end,
            up_sample_start,
            up_sample_end,
            up_sample_size,
        }
    }
}

/// Center evaluator: evaluates B-spline basis functions at cell centers.
pub struct CenterEvaluator {
    pub depth: u32,
    pub degree: usize,
    /// values[offset][cell_offset - support_start] = phi_offset(center_of_cell)
    pub values: Vec<Vec<f64>>,
}

impl CenterEvaluator {
    pub fn new(depth: u32, degree: usize) -> Self {
        let res = 1usize << depth;
        let support = BSplineSupport::new(degree);
        let components = bspline_components(degree);
        let num_offsets = res;

        let mut values = vec![vec![0.0f64; support.support_size]; num_offsets];

        for offset in 0..num_offsets {
            for j in support.support_start..=support.support_end {
                let cell = offset as isize + j;
                if cell < 0 || cell >= res as isize {
                    continue;
                }
                let t = 0.5f64; // cell center
                let comp_idx = (j - support.support_start) as usize;
                if comp_idx < components.len() {
                    values[offset][comp_idx] = components[comp_idx].eval(t);
                }
            }
        }

        CenterEvaluator {
            depth,
            degree,
            values,
        }
    }

    /// Get the value of basis function centered at `f_idx`
    /// evaluated at the center of cell `c_idx`.
    pub fn value(&self, f_idx: usize, c_idx: usize) -> f64 {
        let res = 1usize << self.depth;
        if c_idx >= res || f_idx >= res {
            return 0.0;
        }
        let dd = c_idx as isize - f_idx as isize;
        let support = BSplineSupport::new(self.degree);
        if dd < support.support_start || dd > support.support_end {
            return 0.0;
        }
        let idx = (dd - support.support_start) as usize;
        self.values[f_idx][idx]
    }
}

/// Up-sampling evaluator: prolongation stencil from coarse to fine.
pub struct UpSampleEvaluator {
    pub low_depth: u32,
    pub degree: usize,
    pub values: Vec<Vec<f64>>,
    pub up_sample_start: isize,
    pub up_sample_size: usize,
}

impl UpSampleEvaluator {
    pub fn new(low_depth: u32, degree: usize) -> Self {
        let fine_res = 1usize << (low_depth + 1);
        let num_parents = 1usize << low_depth;
        let support = BSplineSupport::new(degree);
        let components = bspline_components(degree);

        let up_sample_start = support.up_sample_start;
        let up_sample_end = support.up_sample_end;
        let up_sample_size = support.up_sample_size;

        let mut values = vec![vec![0.0f64; up_sample_size]; num_parents];

        for p in 0..num_parents {
            for j in up_sample_start..=up_sample_end {
                let c = 2 * p as isize + j;
                if c < 0 || c >= fine_res as isize {
                    continue;
                }
                let s_coarse = (c as f64 + 0.5) / fine_res as f64;
                let coarse_res_f = (1usize << low_depth) as f64;
                let coarse_cell = (s_coarse * coarse_res_f).floor() as isize;
                let t = s_coarse * coarse_res_f - coarse_cell as f64;

                let comp_idx = (coarse_cell - p as isize - support.support_start) as usize;
                if comp_idx < components.len() {
                    let idx = (j - up_sample_start) as usize;
                    values[p][idx] = components[comp_idx].eval(t);
                }
            }
        }

        UpSampleEvaluator {
            low_depth,
            degree,
            values,
            up_sample_start,
            up_sample_size,
        }
    }

    pub fn value(&self, p_idx: usize, c_idx: usize) -> f64 {
        let dd = c_idx as isize - 2 * p_idx as isize;
        if dd < self.up_sample_start || dd >= self.up_sample_start + self.up_sample_size as isize {
            return 0.0;
        }
        self.values[p_idx][(dd - self.up_sample_start) as usize]
    }
}

/// Evaluate the d-th derivative of a degree-D B-spline centered at offset `off`
/// at position x in [0, 1] (resolution is implicit).
pub fn eval_bspline_derivative(degree: usize, deriv: usize, offset: isize, x: f64) -> f64 {
    let d_plus_1 = degree as isize + 1;
    let shift = d_plus_1 as f64 / 2.0;
    let arg = x - offset as f64 + shift;
    eval_canonical_bspline_derivative(degree, deriv, arg)
}

/// Evaluate the canonical B-spline N_D(x).
pub fn eval_canonical_bspline(degree: usize, x: f64) -> f64 {
    if x < 0.0 || x > (degree + 1) as f64 {
        return 0.0;
    }

    match degree {
        0 => {
            if x >= 0.0 && x < 1.0 { 1.0 } else { 0.0 }
        }
        1 => {
            if x <= 1.0 { x } else { 2.0 - x }
        }
        2 => {
            if x <= 1.0 {
                x * x / 2.0
            } else if x <= 2.0 {
                (-2.0 * x * x + 6.0 * x - 3.0) / 2.0
            } else {
                let t = 3.0 - x;
                t * t / 2.0
            }
        }
        _ => {
            let a = x / degree as f64;
            let b = (degree + 1) as f64 - x;
            a * eval_canonical_bspline(degree - 1, x)
                + (b / degree as f64) * eval_canonical_bspline(degree - 1, x - 1.0)
        }
    }
}

/// Evaluate the d-th derivative of the canonical B-spline.
pub fn eval_canonical_bspline_derivative(degree: usize, deriv: usize, x: f64) -> f64 {
    if deriv == 0 {
        return eval_canonical_bspline(degree, x);
    }
    if deriv > degree {
        return 0.0;
    }
    eval_canonical_bspline_derivative(degree - 1, deriv - 1, x)
        - eval_canonical_bspline_derivative(degree - 1, deriv - 1, x - 1.0)
}

/// Pre-computed dot product matrix for <D^{d1} B_i, D^{d2} B_j>.
/// This is a symmetric Toeplitz matrix: M[|j-i|] = dot product.
/// Returns (values, offset_range) where values[k] corresponds to offset = support_start + k.
pub fn bspline_dot_product(degree1: usize, degree2: usize, d1: usize, d2: usize) -> (Vec<f64>, isize) {
    let max_offset = degree1 + degree2 + 2;
    let support_start = -(max_offset as isize) / 2;
    let support_end = max_offset as isize / 2;
    let n = (support_end - support_start + 1) as usize;

    let samples = 1000;
    let h = 1.0 / samples as f64;
    let mut result = vec![0.0f64; n];

    for k in 0..n {
        let offset = support_start + k as isize;
        let mut dot = 0.0f64;

        for s in 0..=samples {
            let x = s as f64 / samples as f64;
            let v1 = eval_bspline_derivative(degree1, d1, 0isize, x);
            let v2 = eval_bspline_derivative(degree2, d2, offset, x);
            let w = if s == 0 || s == samples { 0.5 } else { 1.0 };
            dot += v1 * v2 * w * h;
        }
        result[k] = dot;
    }

    (result, support_start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bspline_degree0() {
        assert!((eval_canonical_bspline(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((eval_canonical_bspline(0, -0.1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_degree1() {
        assert!((eval_canonical_bspline(1, 0.5) - 0.5).abs() < 1e-10);
        assert!((eval_canonical_bspline(1, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bspline_degree2() {
        let v = eval_canonical_bspline(2, 0.5);
        assert!((v - 0.125).abs() < 1e-10, "N_2(0.5) = {}", v);
        let v = eval_canonical_bspline(2, 1.5);
        assert!((v - 0.75).abs() < 1e-10, "N_2(1.5) = {}", v);
    }

    #[test]
    fn test_bspline_components_degree1() {
        let comps = bspline_components(1);
        assert!((comps[0].eval(0.0) - 1.0).abs() < 1e-10);
        assert!((comps[0].eval(1.0) - 0.0).abs() < 1e-10);
        assert!((comps[1].eval(0.0) - 0.0).abs() < 1e-10);
        assert!((comps[1].eval(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_partition_of_unity() {
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut sum = 0.0;
            for k in -1..=2isize {
                sum += eval_bspline_derivative(1, 0, k, x);
            }
            assert!((sum - 1.0).abs() < 1e-10, "sum at x={}: {}", x, sum);
        }
    }

    #[test]
    fn test_center_evaluator() {
        let eval = CenterEvaluator::new(2, 1); // depth 2 = 4 cells
        // At depth 2, resolution = 4. Cells: 0,1,2,3
        // B-spline of degree 1 centered at offset 1:
        // phi_1 evaluated at cell center 1.5 should be non-zero
        let v = eval.value(1, 1);
        assert!(v > 0.0);
        // phi_1 evaluated at cell center 3 should be 0
        let v = eval.value(1, 3);
        assert!((v - 0.0).abs() < 1e-10);
    }
}
