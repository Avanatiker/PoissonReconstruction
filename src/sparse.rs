/// Sparse matrix in CSR-like format.
///
/// Each row stores a contiguous array of `(column_index, value)` entries.
/// Supports variable-length rows (`MaxRowSize = 0`) and fixed-length rows.


/// A single entry in a sparse matrix row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixEntry<T> {
    pub col: usize,
    pub value: T,
}

impl<T> MatrixEntry<T> {
    #[inline]
    pub fn new(col: usize, value: T) -> Self {
        MatrixEntry { col, value }
    }
}

// Renamed to avoid unused warning — used in multigrid assembly
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SparseMatrix<T> {
    /// Number of rows
    pub rows: usize,
    /// Flattened entries: row i uses entries[offsets[i]..offsets[i+1]]
    entries: Vec<MatrixEntry<T>>,
    /// Offsets into entries for each row
    offsets: Vec<usize>,
    /// Per-row entry counts
    row_sizes: Vec<usize>,
}

impl<T: Clone + Default + std::ops::AddAssign + std::ops::MulAssign + Copy + std::ops::Mul<Output = T> + num_traits::One> SparseMatrix<T> {
    pub fn new(rows: usize) -> Self {
        SparseMatrix {
            rows,
            entries: Vec::new(),
            offsets: vec![0; rows + 1],
            row_sizes: vec![0; rows],
        }
    }

    pub fn with_capacity(rows: usize, nnz: usize) -> Self {
        SparseMatrix {
            rows,
            entries: Vec::with_capacity(nnz),
            offsets: vec![0; rows + 1],
            row_sizes: vec![0; rows],
        }
    }

    /// Set the number of entries in a row. Must be called before populating entries.
    pub fn set_row_size(&mut self, row: usize, size: usize) {
        self.row_sizes[row] = size;
    }

    /// After setting all row sizes, call this to build offsets and allocate entries.
    pub fn finalize_structure(&mut self) {
        let mut offset = 0;
        for i in 0..self.rows {
            self.offsets[i] = offset;
            offset += self.row_sizes[i];
        }
        self.offsets[self.rows] = offset;
        self.entries.resize(offset, MatrixEntry::new(0, T::default()));
    }

    /// Get mutable reference to a row's entries (after finalize_structure).
    #[inline]
    pub fn row_mut(&mut self, row: usize) -> &mut [MatrixEntry<T>] {
        let start = self.offsets[row];
        let end = self.offsets[row + 1];
        &mut self.entries[start..end]
    }

    /// Get immutable reference to a row's entries.
    #[inline]
    pub fn row(&self, row: usize) -> &[MatrixEntry<T>] {
        let start = self.offsets[row];
        let end = self.offsets[row + 1];
        &self.entries[start..end]
    }

    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// Size of a specific row.
    #[inline]
    pub fn row_size(&self, row: usize) -> usize {
        self.row_sizes[row]
    }

    /// Total number of non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Matrix-vector multiply: out = self * input (parallel with rayon)
    pub fn multiply_vector(&self, input: &[T], output: &mut [T])
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Send + Sync + Default + Copy,
    {
        use rayon::prelude::*;
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut sum = T::default();
            for entry in self.row(i) {
                sum = sum + entry.value * input[entry.col];
            }
            *out = sum;
        });
    }

    /// Transpose times vector: out = self^T * input
    pub fn multiply_transpose_vector(&self, input: &[T], output: &mut [T]) {
        output.fill(T::default());
        for i in 0..self.rows {
            let val = input[i];
            for entry in self.row(i) {
                output[entry.col] += entry.value * val;
            }
        }
    }

    /// Scale all entries by a scalar.
    pub fn scale(&mut self, s: T) {
        for entry in &mut self.entries {
            entry.value *= s;
        }
    }

    /// Build an identity matrix.
    pub fn identity(dim: usize) -> Self {
        let mut m = SparseMatrix::new(dim);
        for i in 0..dim {
            m.set_row_size(i, 1);
        }
        m.finalize_structure();
        for i in 0..dim {
            m.row_mut(i)[0] = MatrixEntry::new(i, T::one());
        }
        m
    }
}

impl<T: Copy> std::ops::Mul<T> for SparseMatrix<T>
where
    T: Clone
        + Default
        + std::ops::AddAssign
        + std::ops::MulAssign
        + Copy
        + std::ops::Mul<Output = T>
        + num_traits::One,
{
    type Output = SparseMatrix<T>;

    fn mul(self, s: T) -> SparseMatrix<T> {
        let mut result = self.clone();
        result.scale(s);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_multiply() {
        let mut m: SparseMatrix<f64> = SparseMatrix::new(3);
        m.set_row_size(0, 2); // row 0: col 0, col 1
        m.set_row_size(1, 1); // row 1: col 1
        m.set_row_size(2, 1); // row 2: col 2
        m.finalize_structure();

        m.row_mut(0)[0] = MatrixEntry::new(0, 2.0);
        m.row_mut(0)[1] = MatrixEntry::new(1, 1.0);
        m.row_mut(1)[0] = MatrixEntry::new(1, 3.0);
        m.row_mut(2)[0] = MatrixEntry::new(2, 4.0);

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        m.multiply_vector(&x, &mut y);
        // row 0: 2*1 + 1*2 = 4
        // row 1: 3*2 = 6
        // row 2: 4*3 = 12
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 12.0).abs() < 1e-10);
    }
}
