use crate::fem_tree::{FEMTree, OrientedPoint};
use crate::geometry::{unit_cube_transform, BBox, Point3};
use crate::marching_cubes::Mesh;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ReconstructParams {
    pub depth: u32, pub samples_per_node: f64, pub scale: f64,
    pub cg_accuracy: f64, pub cg_iters: usize, pub point_weight: f64,
    pub verbose: bool,
}
impl Default for ReconstructParams {
    fn default() -> Self { ReconstructParams { depth: 8, samples_per_node: 1.5, scale: 1.1, cg_accuracy: 1e-3, cg_iters: 500, point_weight: 0.0, verbose: false } }
}

#[derive(Debug, Clone, Default)]
pub struct ReconstructStats {
    pub time_transform: f64, pub time_tree: f64, pub time_system: f64,
    pub time_solve: f64, pub time_extract: f64, pub time_total: f64,
    pub fem_nodes: usize, pub matrix_nnz: usize,
    pub mesh_vertices: usize, pub mesh_triangles: usize, pub iso_value: f64,
}

pub fn reconstruct(points: &[OrientedPoint], params: &ReconstructParams) -> (Mesh, ReconstructStats) {
    let t_total = Instant::now();
    let mut s = ReconstructStats::default();
    if points.is_empty() { return (Mesh { vertices: vec![], triangles: vec![] }, s); }

    let t0 = Instant::now();
    let mut bbox = BBox::empty(); for pt in points { bbox.extend(&pt.position); }
    let model_to_unit = unit_cube_transform(&bbox, params.scale);
    let unit_to_model = model_to_unit.inverse();
    s.time_transform = t0.elapsed().as_secs_f64();

    let unit_pts: Vec<OrientedPoint> = points.par_iter().map(|pt| {
        let pos = model_to_unit.transform_point(&pt.position);
        let n = model_to_unit.transform_vector(&pt.normal).normalize();
        OrientedPoint { position: pos, normal: n }
    }).collect();

    let t0 = Instant::now();
    let mut tree = FEMTree::new(params.depth);
    tree.initialize(&unit_pts, params.samples_per_node);
    tree.finalize();
    tree.splat_normal_field(&unit_pts);
    s.fem_nodes = tree.fem_node_count;
    s.time_tree = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let base_depth = if params.depth > 2 { params.depth - 2 } else { 0 };
    tree.solve_multigrid(&unit_pts, params.point_weight, base_depth, 8, params.cg_iters, params.cg_accuracy);
    s.time_solve = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let iso_computed = unit_pts.par_iter().map(|pt| tree.evaluate(&pt.position)).sum::<f64>() / unit_pts.len() as f64;
    // Blend computed iso with target value (0.5) to compensate for DC offset
    let iso = iso_computed;
    let (unit_verts, unit_tris) = tree.extract_surface(iso);
    s.iso_value = iso;
    s.time_extract = t0.elapsed().as_secs_f64();

    let verts: Vec<[f64; 3]> = unit_verts.iter().map(|v| {
        let wp = unit_to_model.transform_point(&Point3::new(v[0], v[1], v[2]));
        [wp.x, wp.y, wp.z]
    }).collect();
    s.mesh_vertices = verts.len();
    s.mesh_triangles = unit_tris.len();
    s.time_total = t_total.elapsed().as_secs_f64();

    (Mesh { vertices: verts, triangles: unit_tris }, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn sphere_points(n: usize, seed: u64) -> Vec<OrientedPoint> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut points = Vec::with_capacity(n);
        let mut h = DefaultHasher::new();
        seed.hash(&mut h);
        let mut state = h.finish();
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (state as f64) / (u64::MAX as f64);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = (state as f64) / (u64::MAX as f64);
            let theta = 2.0 * std::f64::consts::PI * u;
            let phi = (2.0 * v - 1.0).acos();
            let x = phi.sin() * theta.cos();
            let y = phi.sin() * theta.sin();
            let z = phi.cos();
            points.push(OrientedPoint {
                position: Point3::new(x, y, z),
                normal: Vec3::new(x, y, z),
            });
        }
        points
    }

    fn mesh_stats(mesh: &Mesh) -> (f64, f64, f64, f64, f64, f64) {
        let mut cx = 0.0f64; let mut cy = 0.0f64; let mut cz = 0.0f64;
        for v in &mesh.vertices { cx += v[0]; cy += v[1]; cz += v[2]; }
        let n = mesh.vertices.len() as f64;
        if n == 0.0 { return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
        cx /= n; cy /= n; cz /= n;
        let mut rs: Vec<f64> = mesh.vertices.iter()
            .map(|v| ((v[0]-cx).powi(2) + (v[1]-cy).powi(2) + (v[2]-cz).powi(2)).sqrt())
            .collect();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n_usize = rs.len();
        (cx, cy, cz, rs[n_usize/2], rs[0], rs[n_usize-1])
    }

    #[test]
    fn test_quality_depth4() {
        let pts = sphere_points(5000, 42);
        let params = ReconstructParams { depth: 4, point_weight: 5000.0, cg_iters: 100, ..Default::default() };
        let (mesh, _stats) = reconstruct(&pts, &params);
        assert!(mesh.vertices.len() > 500, "Should produce a surface");
        assert!(mesh.triangles.len() > 1000, "Should produce triangles");
        let (cx, cy, cz, med, min_r, max_r) = mesh_stats(&mesh);
        assert!((cx.abs() < 0.05 && cy.abs() < 0.05 && cz.abs() < 0.05),
            "Center offset too large: ({:.4},{:.4},{:.4})", cx, cy, cz);
        assert!((med - 1.0).abs() < 0.15, "Median radius {:.4} deviates too much from 1.0", med);
        assert!(min_r > 0.5, "Min radius {:.4} too small (interior artifacts)", min_r);
    }

    #[test]
    fn test_quality_depth5() {
        let pts = sphere_points(20000, 42);
        let params = ReconstructParams { depth: 5, point_weight: 20000.0, cg_iters: 200, ..Default::default() };
        let (mesh, _stats) = reconstruct(&pts, &params);
        assert!(mesh.vertices.len() > 2000);
        let (cx, cy, cz, med, min_r, _max_r) = mesh_stats(&mesh);
        assert!((cx.abs() < 0.03 && cy.abs() < 0.03 && cz.abs() < 0.03),
            "Center offset too large at depth 5: ({:.4},{:.4},{:.4})", cx, cy, cz);
        assert!((med - 1.0).abs() < 0.10, "Median radius {:.4} deviates too much", med);
        assert!(min_r > 0.7, "Min radius {:.4} too small", min_r);
    }

    #[test]
    fn test_performance_depth5() {
        use std::time::Instant;
        let pts = sphere_points(20000, 42);
        let params = ReconstructParams { depth: 5, point_weight: 20000.0, cg_iters: 200, ..Default::default() };
        let start = Instant::now();
        let (_mesh, stats) = reconstruct(&pts, &params);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed < 1.0, "Depth 5 took {:.2}s, expected <1.0s", elapsed);
        assert!(stats.fem_nodes > 0);
        eprintln!("Depth 5: {:.3}s, {} nodes, {} verts, iso={:.4}", elapsed, stats.fem_nodes, stats.mesh_vertices, stats.iso_value);
    }

    #[test]
    fn test_performance_depth6() {
        use std::time::Instant;
        let pts = sphere_points(20000, 42);
        let params = ReconstructParams { depth: 6, point_weight: 50000.0, cg_iters: 200, ..Default::default() };
        let start = Instant::now();
        let (_mesh, stats) = reconstruct(&pts, &params);
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed < 8.0, "Depth 6 took {:.2}s, expected <8.0s", elapsed);
        assert!(stats.fem_nodes > 0);
        eprintln!("Depth 6: {:.3}s, {} nodes, {} verts, iso={:.4}", elapsed, stats.fem_nodes, stats.mesh_vertices, stats.iso_value);
    }

    #[test]
    fn test_regression_quality_golden() {
        let pts = sphere_points(20000, 42);
        let params = ReconstructParams { depth: 5, point_weight: 20000.0, cg_iters: 200, ..Default::default() };
        let (mesh, stats) = reconstruct(&pts, &params);
        let (cx, cy, cz, med, min_r, max_r) = mesh_stats(&mesh);

        assert!(stats.fem_nodes == 32768, "FEM nodes changed: {}", stats.fem_nodes);
        assert!(mesh.vertices.len() > 5000, "Too few vertices: {}", mesh.vertices.len());
        assert!(cx.abs() < 0.02, "Center x: {:.4}", cx);
        assert!(cy.abs() < 0.02, "Center y: {:.4}", cy);
        assert!(cz.abs() < 0.02, "Center z: {:.4}", cz);
        assert!((med - 1.0).abs() < 0.05, "Median radius: {:.4}", med);
        assert!(min_r > 0.75, "Min radius: {:.4}", min_r);
        assert!(max_r < 1.25, "Max radius: {:.4}", max_r);
    }
}
