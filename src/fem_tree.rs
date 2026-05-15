use std::collections::HashMap;
use crate::geometry::{Point3, Vec3};
use crate::octree::{self, OctreeNode};
use crate::sparse::{MatrixEntry, SparseMatrix};
use crate::solvers;

pub const DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct OrientedPoint { pub position: Point3, pub normal: Vec3 }

// Standard MC tables
const EDGE_TABLE: [u16; 256] = [
    0x000,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x099,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x033,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0x0aa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x066,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0x0ff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x055,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0x0cc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0x0cc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x055,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0x0ff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x066,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0x0aa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x033,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x099,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x000,
];

// The first 24 entries are sufficient for the common MC configurations.
// For brevity, we fill the rest with the full table from the mcubes crate.
// This is a compact version of the full 256-entry table.
include!("tri_table.inl");


/// 1D B-spline stencil constants for degree 1.
fn mass_1d(d: isize) -> f64 { match d.abs() { 0 => 2./3., 1 => 1./6., _ => 0.0 } }
fn stiff_1d(d: isize) -> f64 { match d.abs() { 0 => 2., 1 => -1., _ => 0.0 } }
fn deriv_mass_1d(d: isize) -> f64 { match d { 0 => 0., 1 => -0.5, -1 => 0.5, _ => 0.0 } }

/// Evaluate the 3D Laplacian stiffness entry between FEM nodes at (ix,iy,iz) and (jx,jy,jz).
fn stiffness_entry(h: f64, dx: isize, dy: isize, dz: isize) -> f64 {
    (stiff_1d(dx) * mass_1d(dy) * mass_1d(dz)
   + mass_1d(dx) * stiff_1d(dy) * mass_1d(dz)
   + mass_1d(dx) * mass_1d(dy) * stiff_1d(dz)) * h
}

/// N_1(t) — the canonical degree-1 B-spline.
fn bspline_val(s: f64, offset: u32) -> f64 {
    let d = s - offset as f64;
    if d < 0.0 || d > 2.0 { 0.0 } else if d <= 1.0 { d } else { 2.0 - d }
}

pub struct FEMTree {
    pub max_depth: u32,
    pub fem_node_count: usize,
    pub octree: octree::Octree,
    /// Dense grid mapping: grid_to_idx[z*res*res + y*res + x] = FEM index (usize::MAX if none)
    pub grid_to_idx: Vec<usize>,
    pub idx_to_offset: Vec<[u32; DIM]>,
    pub solution: Vec<f64>,
    pub normal_field: Vec<Vec3>,
    pub node_weight: Vec<f64>,
}

impl FEMTree {
    pub fn new(max_depth: u32) -> Self {
        FEMTree { max_depth, fem_node_count: 0, octree: octree::Octree::new(),
            grid_to_idx: Vec::new(), idx_to_offset: Vec::new(),
            solution: Vec::new(), normal_field: Vec::new(), node_weight: Vec::new() }
    }

    /// Fast O(1) lookup: offset → FEM index.
    #[inline]
    fn idx_at(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        let res = 1usize << self.max_depth;
        if x >= res || y >= res || z >= res { return None; }
        let i = self.grid_to_idx.get(z * res * res + y * res + x).copied().unwrap_or(usize::MAX);
        if i == usize::MAX { None } else { Some(i) }
    }

    pub fn initialize(&mut self, _points: &[OrientedPoint], _threshold: f64) {
        Self::refine_uniform(&mut self.octree.root, self.max_depth);
        self.octree.max_depth = self.max_depth;
    }

    fn refine_uniform(node: &mut OctreeNode, target: u32) {
        if node.depth < target {
            node.init_children();
            if let Some(ref mut kids) = node.children {
                for c in kids.iter_mut() { Self::refine_uniform(c, target); }
            }
        }
    }

    pub fn finalize(&mut self) {
        self.octree.finalize();
        let sorted = self.octree.sorted_nodes.as_ref().expect("not finalized");
        self.idx_to_offset.clear();
        let res = 1usize << self.max_depth;
        self.grid_to_idx = vec![usize::MAX; res * res * res];
        let mut idx = 0;
        for &np in &sorted.tree_nodes {
            unsafe {
                let n = &mut *np;
                if n.is_leaf() && n.depth == self.max_depth {
                    n.data_mut().set_flag(octree::flags::SPACE_FLAG, true);
                    n.data_mut().set_flag(octree::flags::FEM_FLAG_1, true);
                    n.data_mut().node_index = idx as octree::NodeIndex;
                    let o = n.offset;
                    self.grid_to_idx[o[2] as usize * res * res + o[1] as usize * res + o[0] as usize] = idx;
                    self.idx_to_offset.push(o); idx += 1;
                }
            }
        }
        self.fem_node_count = idx; self.solution.resize(idx, 0.0);
    }

    pub fn splat_normal_field(&mut self, points: &[OrientedPoint]) {
        let n = self.fem_node_count;
        self.normal_field.resize(n, Vec3::ZERO);
        self.node_weight.resize(n, 0.0);
        let res = 1usize << self.max_depth; let h = 1.0 / res as f64;
        for pt in points {
            let cx = (pt.position.x / h) as isize; let cy = (pt.position.y / h) as isize; let cz = (pt.position.z / h) as isize;
            let tx = pt.position.x / h - cx as f64; let ty = pt.position.y / h - cy as f64; let tz = pt.position.z / h - cz as f64;
            let bx0 = 1.0 - tx; let bx1 = tx; let by0 = 1.0 - ty; let by1 = ty; let bz0 = 1.0 - tz; let bz1 = tz;
            for dz_i in 0..=1isize { for dy_i in 0..=1isize { for dx_i in 0..=1isize {
                let ox = (cx + dx_i) as u32; let oy = (cy + dy_i) as u32; let oz = (cz + dz_i) as u32;
                if ox >= res as u32 || oy >= res as u32 || oz >= res as u32 { continue; }
                if let Some(idx) = self.idx_at(ox as usize, oy as usize, oz as usize) {
                    let bx = if dx_i == 0 { bx0 } else { bx1 };
                    let by = if dy_i == 0 { by0 } else { by1 };
                    let bz = if dz_i == 0 { bz0 } else { bz1 };
                    let w = bx * by * bz;
                    if idx < n {
                        self.normal_field[idx].x += pt.normal.x * w;
                        self.normal_field[idx].y += pt.normal.y * w;
                        self.normal_field[idx].z += pt.normal.z * w;
                        self.node_weight[idx] += w;
                    }
                }
            }}}
        }
        // Normalize: divide normal by weight, then negate (Poisson convention)
        for i in 0..n {
            if self.node_weight[i] > 0.0 {
                let inv = 1.0 / self.node_weight[i];
                self.normal_field[i].x = -self.normal_field[i].x * inv;
                self.normal_field[i].y = -self.normal_field[i].y * inv;
                self.normal_field[i].z = -self.normal_field[i].z * inv;
            }
        }
    }

    pub fn assemble_system(&self, points: &[OrientedPoint], pw: f64) -> SparseMatrix<f64> {
        let n = self.fem_node_count;
        let res = 1usize << self.max_depth;
        let h = 1.0 / res as f64;
        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        // Laplacian stiffness
        for i in 0..n {
            let [ix, iy, iz] = self.idx_to_offset[i];
            for dx in -1isize..=1isize { for dy in -1isize..=1isize { for dz in -1isize..=1isize {
                let jx = ix as isize + dx; let jy = iy as isize + dy; let jz = iz as isize + dz;
                if jx < 0 || jx >= res as isize || jy < 0 || jy >= res as isize || jz < 0 || jz >= res as isize { continue; }
                if let Some(j) = self.idx_at(jx as usize, jy as usize, jz as usize) {
                    let val = stiffness_entry(h, dx, dy, dz);
                    if val != 0.0 { rows[i].push((j, val)); }
                }
            }}}
        }

        // Point interpolation mass (screening) — full 8x8 sparse block per sample
        if pw > 0.0 {
            let wh3 = pw * h.powi(3);
            for pt in points {
                let cx = (pt.position.x / h) as isize; let cy = (pt.position.y / h) as isize; let cz = (pt.position.z / h) as isize;
                let tx = pt.position.x / h - cx as f64; let ty = pt.position.y / h - cy as f64; let tz = pt.position.z / h - cz as f64;
                let bx0 = 1. - tx; let bx1 = tx; let by0 = 1. - ty; let by1 = ty; let bz0 = 1. - tz; let bz1 = tz;
                let mut basis: Vec<(usize, f64)> = Vec::with_capacity(8);
                for dz_i in 0..=1isize { for dy_i in 0..=1isize { for dx_i in 0..=1isize {
                    let ox = (cx + dx_i) as u32; let oy = (cy + dy_i) as u32; let oz = (cz + dz_i) as u32;
                    if ox >= res as u32 || oy >= res as u32 || oz >= res as u32 { continue; }
                    if let Some(idx) = self.idx_at(ox as usize, oy as usize, oz as usize) {
                        let bx = if dx_i == 0 { bx0 } else { bx1 };
                        let by = if dy_i == 0 { by0 } else { by1 };
                        let bz = if dz_i == 0 { bz0 } else { bz1 };
                        basis.push((idx, bx * by * bz));
                    }
                }}}
                for &(i_idx, phi_i) in &basis {
                    for &(j_idx, phi_j) in &basis {
                        rows[i_idx].push((j_idx, wh3 * phi_i * phi_j));
                    }
                }
            }
        }

        // Deduplicate and build sparse matrix
        for row in &mut rows {
            row.sort_by_key(|(j, _)| *j);
            let mut j = 0; let mut u = 0;
            while j < row.len() {
                let col = row[j].0; let mut sum = 0.0;
                while j < row.len() && row[j].0 == col { sum += row[j].1; j += 1; }
                row[u] = (col, sum); u += 1;
            }
            row.truncate(u);
        }
        let nnz: usize = rows.iter().map(|r| r.len()).sum();
        let mut mat = SparseMatrix::with_capacity(n, nnz);
        for i in 0..n { mat.set_row_size(i, rows[i].len()); }
        mat.finalize_structure();
        for i in 0..n {
            let row = mat.row_mut(i);
            for (k, &(col, val)) in rows[i].iter().enumerate() { row[k] = MatrixEntry::new(col, val); }
        }
        mat
    }

    pub fn assemble_rhs(&self, points: &[OrientedPoint], pw: f64) -> Vec<f64> {
        let n = self.fem_node_count;
        let res = 1usize << self.max_depth;
        let h = 1.0 / res as f64;
        let mut rhs = vec![0.0; n];

        // Laplacian RHS from normal field
        for i in 0..n {
            let [ix, iy, iz] = self.idx_to_offset[i];
            let mut sum = 0.0;
            for dx in -1isize..=1isize { for dy in -1isize..=1isize { for dz in -1isize..=1isize {
                let jx = ix as isize + dx; let jy = iy as isize + dy; let jz = iz as isize + dz;
                if jx < 0 || jx >= res as isize || jy < 0 || jy >= res as isize || jz < 0 || jz >= res as isize { continue; }
                if let Some(j) = self.idx_at(jx as usize, jy as usize, jz as usize) {
                    if j < self.normal_field.len() {
                        let v = self.normal_field[j];
                        sum -= v.x * deriv_mass_1d(dx) * mass_1d(dy) * mass_1d(dz)
                             + v.y * mass_1d(dx) * deriv_mass_1d(dy) * mass_1d(dz)
                             + v.z * mass_1d(dx) * mass_1d(dy) * deriv_mass_1d(dz);
                    }
                }
            }}}
            rhs[i] = sum;
        }

        // Screening RHS
        if pw > 0.0 {
            let wh3 = pw * h.powi(3);
            let target = 0.5;
            for pt in points {
                let cx = (pt.position.x / h) as isize; let cy = (pt.position.y / h) as isize; let cz = (pt.position.z / h) as isize;
                let tx = pt.position.x / h - cx as f64; let ty = pt.position.y / h - cy as f64; let tz = pt.position.z / h - cz as f64;
                let bx0 = 1. - tx; let bx1 = tx; let by0 = 1. - ty; let by1 = ty; let bz0 = 1. - tz; let bz1 = tz;
                for dz_i in 0..=1isize { for dy_i in 0..=1isize { for dx_i in 0..=1isize {
                    let ox = (cx + dx_i) as u32; let oy = (cy + dy_i) as u32; let oz = (cz + dz_i) as u32;
                    if ox >= res as u32 || oy >= res as u32 || oz >= res as u32 { continue; }
                    if let Some(idx) = self.idx_at(ox as usize, oy as usize, oz as usize) {
                        let bx = if dx_i == 0 { bx0 } else { bx1 };
                        let by = if dy_i == 0 { by0 } else { by1 };
                        let bz = if dz_i == 0 { bz0 } else { bz1 };
                        rhs[idx] += wh3 * target * bx * by * bz;
                    }
                }}}
            }
        }
        rhs
    }

    pub fn solve(&mut self, a: &SparseMatrix<f64>, b: &[f64], gs_iters: usize, cg_iters: usize, eps: f64) {
        let n = a.num_rows();
        self.solution.resize(n, 0.0);
        self.solution.fill(0.0);
        for _ in 0..gs_iters { solvers::gauss_seidel_sweep(a, b, &mut self.solution); }
        solvers::solve_cg(a, b, &mut self.solution, cg_iters, eps);
    }

    /// Prolongate solution from coarse to fine using degree-1 B-spline up-sampling.
    fn prolongate(&self, coarse: &[f64], coarse_depth: u32) -> Vec<f64> {
        let fine_res = 1usize << self.max_depth;
        let coarse_res = 1usize << coarse_depth;
        let mut fine = vec![0.0; self.fem_node_count];

        let bw = [0.5, 1.0, 0.5];
        for cz in 0..coarse_res {
            for cy in 0..coarse_res {
                for cx in 0..coarse_res {
                    let ci = (cz * coarse_res + cy) * coarse_res + cx;
                    if ci >= coarse.len() { continue; }
                    let v = coarse[ci];
                    if v == 0.0 { continue; }
                    for dz in 0..=2isize {
                        let fz = cz as isize * 2 + dz - 1;
                        if fz < 0 || fz >= fine_res as isize { continue; }
                        for dy in 0..=2isize {
                            let fy = cy as isize * 2 + dy - 1;
                            if fy < 0 || fy >= fine_res as isize { continue; }
                            for dx in 0..=2isize {
                                let fx = cx as isize * 2 + dx - 1;
                                if fx < 0 || fx >= fine_res as isize { continue; }
                                let w = bw[dx as usize] * bw[dy as usize] * bw[dz as usize];
                                if let Some(fi) = self.idx_at(fx as usize, fy as usize, fz as usize) {
                                    if fi < fine.len() { fine[fi] += v * w; }
                                }
                            }
                        }
                    }
                }
            }
        }
        fine
    }

    /// Build FEM system at a specified depth (regular grid).
    fn build_system_at_depth(&self, depth: u32, pw: f64) -> (SparseMatrix<f64>, Vec<f64>) {
        let res = 1usize << depth;
        let h = 1.0 / res as f64;
        let n = res * res * res;

        // Rescale screening: the ratio mass/stiffness is h² * pw.
        // At depth d, h = 1/2^d. To match the effective screening at max_depth,
        // we need pw * h³ / h = pw * h² to be consistent.
        // Since h_coarse = h_fine * 2^(max_depth - depth),
        // we scale pw by 2^(2*(max_depth - depth)) to get the same mass/stiffness ratio.
        let scale_factor = (1u64 << (2 * (self.max_depth - depth))) as f64;
        let effective_pw = pw * scale_factor;

        let mut rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for z in 0..res { for y in 0..res { for x in 0..res {
            let i = (z * res + y) * res + x;
            for dz in -1isize..=1isize { for dy in -1isize..=1isize { for dx in -1isize..=1isize {
                let jx = x as isize + dx; let jy = y as isize + dy; let jz = z as isize + dz;
                if jx < 0 || jx >= res as isize || jy < 0 || jy >= res as isize || jz < 0 || jz >= res as isize { continue; }
                let j = (jz as usize * res + jy as usize) * res + jx as usize;
                let v = stiffness_entry(h, dx, dy, dz);
                if v != 0.0 { rows[i].push((j, v)); }
            }}}
            if effective_pw > 0.0 { rows[i].push((i, effective_pw * h.powi(3))); }
        }}}

        for row in &mut rows {
            row.sort_by_key(|(j,_)|*j);
            let mut j=0;let mut u=0;
            while j<row.len(){let c=row[j].0;let mut s=0.;while j<row.len()&&row[j].0==c{s+=row[j].1;j+=1;}row[u]=(c,s);u+=1;}
            row.truncate(u);
        }
        let mut mat = SparseMatrix::with_capacity(n, rows.iter().map(|r|r.len()).sum());
        for i in 0..n { mat.set_row_size(i, rows[i].len()); }
        mat.finalize_structure();
        for i in 0..n { let r = mat.row_mut(i); for (k,&(c,v)) in rows[i].iter().enumerate(){r[k]=MatrixEntry::new(c,v);} }

        let mut rhs = vec![0.0; n];
        // Aggregate fine normal field to coarse
        for i in 0..self.fem_node_count {
            let [fx, fy, fz] = self.idx_to_offset[i];
            let cx = fx >> (self.max_depth - depth);
            let cy = fy >> (self.max_depth - depth);
            let cz = fz >> (self.max_depth - depth);
            let ci = (cz as usize * res + cy as usize) * res + cx as usize;
            if ci < n {
                let v = self.normal_field.get(i).copied().unwrap_or(Vec3::ZERO);
                rhs[ci] += v.x + v.y + v.z;
            }
        }
        if effective_pw > 0.0 {
            let wh3 = effective_pw * h.powi(3);
            for i in 0..n { rhs[i] += wh3 * 0.5; }
        }
        (mat, rhs)
    }

    /// Cascadic solver: CG at coarsest level, then GS relaxation up to finest.
    /// Matches C++ approach: cascadic=true, cgDepth=0, iters=8.
    pub fn solve_cascadic(
        &mut self, points: &[OrientedPoint], pw: f64,
        base_depth: u32, gs: usize, cg: usize, eps: f64,
    ) {
        let fine_sys = self.assemble_system(points, pw);
        let fine_rhs = self.assemble_rhs(points, pw);

        if base_depth >= self.max_depth {
            self.solution.resize(self.fem_node_count, 0.0);
            self.solution.fill(0.0);
            for _ in 0..gs { solvers::gauss_seidel_sweep(&fine_sys, &fine_rhs, &mut self.solution); }
            solvers::solve_cg(&fine_sys, &fine_rhs, &mut self.solution, cg, eps);
            return;
        }

        // Build and CG-solve at base_depth
        let (bmat, brhs) = self.build_system_at_depth(base_depth, pw);
        let mut csol = vec![0.0; brhs.len()];
        for _ in 0..gs { solvers::gauss_seidel_sweep(&bmat, &brhs, &mut csol); }
        solvers::solve_cg(&bmat, &brhs, &mut csol, cg, eps);

        // Prolongate to fine as initial guess
        let init = self.prolongate(&csol, base_depth);
        self.solution.resize(self.fem_node_count, 0.0);
        for i in 0..self.fem_node_count.min(init.len()) { self.solution[i] = init[i]; }

        // GS relaxation only at fine level (matches C++ cascadic: no CG at fine level)
        for _ in 0..gs { solvers::gauss_seidel_sweep(&fine_sys, &fine_rhs, &mut self.solution); }
    }

    /// Evaluate FEM implicit function at any point in the unit cube.
    pub fn evaluate(&self, p: &Point3) -> f64 {
        let res = 1usize << self.max_depth;
        let h = 1.0 / res as f64;
        let sx = p.x / h; let sy = p.y / h; let sz = p.z / h;

        let (ox0, ox1) = contributing_offsets(sx, res);
        let (oy0, oy1) = contributing_offsets(sy, res);
        let (oz0, oz1) = contributing_offsets(sz, res);
        let bx0 = bspline_val(sx, ox0); let bx1 = bspline_val(sx, ox1);
        let by0 = bspline_val(sy, oy0); let by1 = bspline_val(sy, oy1);
        let bz0 = bspline_val(sz, oz0); let bz1 = bspline_val(sz, oz1);

        let mut v = 0.0;
        for &(ox, bx) in &[(ox0, bx0), (ox1, bx1)] { if bx == 0.0 { continue; }
            for &(oy, by) in &[(oy0, by0), (oy1, by1)] { if by == 0.0 { continue; }
                for &(oz, bz) in &[(oz0, bz0), (oz1, bz1)] { if bz == 0.0 { continue; }
                    if let Some(idx) = self.idx_at(ox as usize, oy as usize, oz as usize) {
                        if idx < self.solution.len() { v += self.solution[idx] * bx * by * bz; }
                    }
                }
            }
        }
        v
    }

    /// Extract iso-surface by walking all octree leaves with standard MC triTable.
    pub fn extract_surface(&self, iso: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut verts: Vec<[f64; 3]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let mut edge_hash: HashMap<u64, usize> = HashMap::new();

        let sorted = self.octree.sorted_nodes.as_ref().expect("not finalized");
        let max_res = 1u64 << self.max_depth;

        let co: [[u32; 3]; 8] = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
        let ev: [[usize; 2]; 12] = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

        for &node_ptr in &sorted.tree_nodes {
            let node = unsafe { &*node_ptr };
            if !node.is_leaf() { continue; }
            let d = node.depth;
            let off = node.offset;
            let scale = max_res >> d as u64;
            let w = scale as f64 / max_res as f64;

            let mut cv = [0.0f64; 8]; let mut mc = 0usize;
            for k in 0..8 {
                let c = co[k];
                let p = Point3::new(
                    (off[0] as u64 + c[0] as u64) as f64 * w,
                    (off[1] as u64 + c[1] as u64) as f64 * w,
                    (off[2] as u64 + c[2] as u64) as f64 * w,
                );
                cv[k] = self.evaluate(&p);
                if cv[k] < iso { mc |= 1 << k; }
            }

            let em = EDGE_TABLE[mc] as u16;
            if em == 0 { continue; }

            let mut evi = [usize::MAX; 12];
            for e in 0..12 {
                if (em & (1 << e)) == 0 { continue; }
                let arr = ev[e]; let a = arr[0]; let b = arr[1];
                let dv = cv[b] - cv[a];
                let t = if dv.abs() < 1e-15 { 0.5 } else { (iso - cv[a]) / dv };
                let ca = co[a]; let cb = co[b];
                let ax = (off[0] as u64 + ca[0] as u64) as f64 * w;
                let ay = (off[1] as u64 + ca[1] as u64) as f64 * w;
                let az = (off[2] as u64 + ca[2] as u64) as f64 * w;
                let pos = [
                    ax + t * (cb[0] as i32 - ca[0] as i32) as f64 * w,
                    ay + t * (cb[1] as i32 - ca[1] as i32) as f64 * w,
                    az + t * (cb[2] as i32 - ca[2] as i32) as f64 * w,
                ];
                let h = (pos[0] * 1e7) as u64 ^ (pos[1] * 1e7) as u64 ^ (pos[2] * 1e7) as u64;
                if let Some(ex) = edge_hash.get(&h) { evi[e] = *ex; }
                else { let idx = verts.len(); edge_hash.insert(h, idx); verts.push(pos); evi[e] = idx; }
            }

            for i in (0..16).step_by(3) {
                if TRI_TABLE[mc][i] < 0 { break; }
                tris.push([evi[TRI_TABLE[mc][i] as usize], evi[TRI_TABLE[mc][i+1] as usize], evi[TRI_TABLE[mc][i+2] as usize]]);
            }
        }
        (verts, tris)
    }
}

fn contributing_offsets(s: f64, res: usize) -> (u32, u32) {
    let f = s.floor() as isize;
    let r = res as isize;
    ((f - 1).clamp(0, r - 1) as u32, f.clamp(0, r - 1) as u32)
}
