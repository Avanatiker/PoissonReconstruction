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
    tree.solve_cascadic(&unit_pts, params.point_weight, base_depth, 20, params.cg_iters, params.cg_accuracy);
    s.time_solve = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let iso = unit_pts.par_iter().map(|pt| tree.evaluate(&pt.position)).sum::<f64>() / unit_pts.len() as f64;
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
