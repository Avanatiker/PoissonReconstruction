/// Iterative linear solvers: Conjugate Gradient, Preconditioned CG, Gauss-Seidel.

use crate::sparse::SparseMatrix;

/// Compute squared L2 norm of a vector
pub fn square_norm<T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default>(
    v: &[T],
) -> T {
    v.iter().fold(T::default(), |acc, &x| acc + x * x)
}

/// Compute dot product of two vectors
pub fn dot<T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Copy + Default>(
    a: &[T],
    b: &[T],
) -> T {
    a.iter().zip(b.iter()).fold(T::default(), |acc, (&x, &y)| acc + x * y)
}

/// Solve Ax = b using Conjugate Gradient.
///
/// Returns the number of iterations performed. The solution is stored in `x`.
pub fn solve_cg(
    a: &SparseMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    max_iters: usize,
    accuracy: f64,
) -> usize {
    let n = a.num_rows();
    let mut r = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];

    // r = b - A*x
    a.multiply_vector(x, &mut q);
    for i in 0..n {
        r[i] = b[i] - q[i];
    }
    d.copy_from_slice(&r);

    let mut delta_new = dot(&r, &r);
    let delta_0 = delta_new;

    if delta_new == 0.0 {
        return 0;
    }

    for iter in 0..max_iters {
        a.multiply_vector(&d, &mut q);

        let dq = dot(&d, &q);
        if dq == 0.0 {
            return iter + 1;
        }

        let alpha = delta_new / dq;

        for i in 0..n {
            x[i] += alpha * d[i];
        }

        if (iter + 1) % 50 == 0 {
            a.multiply_vector(x, &mut q);
            for i in 0..n {
                r[i] = b[i] - q[i];
            }
        } else {
            for i in 0..n {
                r[i] -= alpha * q[i];
            }
        }

        let delta_old = delta_new;
        delta_new = dot(&r, &r);

        if delta_new <= delta_0 * accuracy * accuracy {
            return iter + 1;
        }

        let beta = delta_new / delta_old;

        for i in 0..n {
            d[i] = r[i] + beta * d[i];
        }
    }

    max_iters
}

/// Solve Ax = b using Preconditioned Conjugate Gradient with diagonal preconditioner.
pub fn solve_pcg(
    a: &SparseMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    max_iters: usize,
    accuracy: f64,
) -> usize {
    let n = a.num_rows();

    // Build diagonal preconditioner
    let mut inv_diag = vec![0.0f64; n];
    for i in 0..n {
        for entry in a.row(i) {
            if entry.col == i {
                inv_diag[i] = entry.value;
            }
        }
        if inv_diag[i] != 0.0 {
            inv_diag[i] = 1.0 / inv_diag[i];
        }
    }

    let mut r = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];

    // r = b - A*x
    a.multiply_vector(x, &mut q);
    for i in 0..n {
        r[i] = b[i] - q[i];
    }

    // d = M^{-1} * r (diagonal preconditioner)
    for i in 0..n {
        d[i] = inv_diag[i] * r[i];
    }

    let mut delta_new = dot(&r, &d);
    let delta_0 = delta_new;

    if delta_new == 0.0 {
        return 0;
    }

    for iter in 0..max_iters {
        a.multiply_vector(&d, &mut q);

        let dq = dot(&d, &q);
        if dq == 0.0 {
            return iter + 1;
        }

        let alpha = delta_new / dq;

        for i in 0..n {
            x[i] += alpha * d[i];
        }

        if (iter + 1) % 50 == 0 {
            a.multiply_vector(x, &mut q);
            for i in 0..n {
                r[i] = b[i] - q[i];
            }
        } else {
            for i in 0..n {
                r[i] -= alpha * q[i];
            }
        }

        for i in 0..n {
            s[i] = inv_diag[i] * r[i];
        }

        let delta_old = delta_new;
        delta_new = dot(&r, &s);

        if delta_new <= delta_0 * accuracy * accuracy {
            return iter + 1;
        }

        let beta = delta_new / delta_old;

        for i in 0..n {
            d[i] = s[i] + beta * d[i];
        }
    }

    max_iters
}

/// Apply Gauss-Seidel relaxation (one forward sweep).
/// Solves (L + D)*x = b - U*x_old, i.e., x_new = D^{-1} * (b - L*x_new - U*x_old).
/// This is a basic Gauss-Seidel for symmetric positive-definite matrices.
pub fn gauss_seidel_sweep(a: &SparseMatrix<f64>, b: &[f64], x: &mut [f64]) {
    let n = a.num_rows();
    for i in 0..n {
        let mut diag = 0.0f64;
        let mut sum = 0.0f64;
        for entry in a.row(i) {
            if entry.col == i {
                diag = entry.value;
            } else {
                sum += entry.value * x[entry.col];
            }
        }
        if diag != 0.0 {
            x[i] = (b[i] - sum) / diag;
        }
    }
}

/// Apply symmetric Gauss-Seidel (forward then backward sweep).
pub fn symmetric_gauss_seidel_sweep(a: &SparseMatrix<f64>, b: &[f64], x: &mut [f64]) {
    let n = a.num_rows();

    // Forward sweep
    for i in 0..n {
        let mut diag = 0.0f64;
        let mut sum = 0.0f64;
        for entry in a.row(i) {
            if entry.col == i {
                diag = entry.value;
            } else {
                sum += entry.value * x[entry.col];
            }
        }
        if diag != 0.0 {
            x[i] = (b[i] - sum) / diag;
        }
    }

    // Backward sweep
    for i in (0..n).rev() {
        let mut diag = 0.0f64;
        let mut sum = 0.0f64;
        for entry in a.row(i) {
            if entry.col == i {
                diag = entry.value;
            } else {
                sum += entry.value * x[entry.col];
            }
        }
        if diag != 0.0 {
            x[i] = (b[i] - sum) / diag;
        }
    }
}

/// Over-relaxed Jacobi smoother.
pub fn jacobi_sweep(
    a: &SparseMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    omega: f64,
    temp: &mut [f64],
) {
    let n = a.num_rows();

    for i in 0..n {
        let mut diag = 0.0f64;
        let mut ax = 0.0f64;
        for entry in a.row(i) {
            if entry.col == i {
                diag = entry.value;
            }
            ax += entry.value * x[entry.col];
        }
        if diag != 0.0 {
            temp[i] = x[i] + omega * (b[i] - ax) / diag;
        } else {
            temp[i] = x[i];
        }
    }

    x.copy_from_slice(temp);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_simple() {
        let n = 10usize;
        let mut a: SparseMatrix<f64> = SparseMatrix::new(n);

        for i in 0..n {
            let size = if i > 0 && i < n - 1 { 3 } else { 2 };
            a.set_row_size(i, size);
        }
        a.finalize_structure();

        for i in 0..n {
            let row = a.row_mut(i);
            let mut k = 0;
            if i > 0 {
                row[k] = crate::sparse::MatrixEntry::new(i - 1, -1.0);
                k += 1;
            }
            row[k] = crate::sparse::MatrixEntry::new(i, 2.0);
            k += 1;
            if i < n - 1 {
                row[k] = crate::sparse::MatrixEntry::new(i + 1, -1.0);
            }
        }

        let mut x = vec![0.0f64; n];
        let b = vec![1.0f64; n];

        solve_cg(&a, &b, &mut x, 100, 1e-10);

        for i in 0..n {
            let expected = ((i + 1) * (n - i)) as f64 / 2.0;
            assert!((x[i] - expected).abs() < 1e-6, "x[{}] = {}, expected {}", i, x[i], expected);
        }
    }
}
