use crate::fem_tree::{FEMTree, OrientedPoint};
use crate::geometry::{unit_cube_transform, BBox, Point3};
use crate::marching_cubes::Mesh;
use rayon::prelude::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ReconstructParams {
    pub depth: u32, pub samples_per_node: f64, pub scale: f64,
    pub cg_accuracy: f64, pub cg_iters: usize, pub point_weight: f64,
    pub grid_resolution: usize, pub verbose: bool,
}
impl Default for ReconstructParams {
    fn default() -> Self {
        ReconstructParams { depth: 8, samples_per_node: 1.5, scale: 1.1, cg_accuracy: 1e-3, cg_iters: 300, point_weight: 0.0, grid_resolution: 64, verbose: false }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReconstructStats {
    pub time_transform: f64, pub time_tree_build: f64, pub time_matrix_assemble: f64,
    pub time_rhs_assemble: f64, pub time_solve: f64, pub time_evaluate: f64,
    pub time_marching_cubes: f64, pub time_total: f64,
    pub fem_nodes: usize, pub matrix_nnz: usize,
    pub mesh_vertices: usize, pub mesh_triangles: usize, pub iso_value: f64,
}

pub fn reconstruct(points: &[OrientedPoint], params: &ReconstructParams) -> (Mesh, ReconstructStats) {
    let t_total = Instant::now();
    let mut stats = ReconstructStats::default();
    if points.is_empty() { return (Mesh { vertices: vec![], triangles: vec![] }, stats); }

    let t0 = Instant::now();
    let mut bbox = BBox::empty(); for pt in points { bbox.extend(&pt.position); }
    let model_to_unit = unit_cube_transform(&bbox, params.scale);
    let unit_to_model = model_to_unit.inverse();
    stats.time_transform = t0.elapsed().as_secs_f64();

    let unit_points: Vec<OrientedPoint> = points.par_iter().map(|pt| {
        let pos = model_to_unit.transform_point(&pt.position);
        let n = model_to_unit.transform_vector(&pt.normal).normalize();
        OrientedPoint { position: pos, normal: n }
    }).collect();

    let t0 = Instant::now();
    let mut tree = FEMTree::new(params.depth);
    tree.initialize_from_points(&unit_points, params.samples_per_node);
    tree.finalize();
    tree.splat_normal_field(&unit_points);
    stats.fem_nodes = tree.fem_node_count;
    stats.time_tree_build = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let stiffness = tree.assemble_system_matrix(&unit_points, params.depth, params.point_weight);
    stats.matrix_nnz = stiffness.nnz();
    stats.time_matrix_assemble = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let rhs = tree.assemble_rhs_from_field(&unit_points, params.depth, params.point_weight);
    stats.time_rhs_assemble = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    tree.solve_cascadic(&stiffness, &rhs, 5, params.cg_iters, params.cg_accuracy);
    stats.time_solve = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    // Evaluate FEM at sample positions to compute iso-value
    let iso = unit_points.par_iter().map(|pt| tree.evaluate(&pt.position, params.depth)).sum::<f64>() / unit_points.len() as f64;
    stats.iso_value = iso;
    stats.time_evaluate = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let (unit_verts, unit_tris) = tree.extract_surface(iso);
    let verts: Vec<[f64; 3]> = unit_verts.iter().map(|v| {
        let wp = unit_to_model.transform_point(&Point3::new(v[0], v[1], v[2]));
        [wp.x, wp.y, wp.z]
    }).collect();
    stats.mesh_vertices = verts.len();
    stats.mesh_triangles = unit_tris.len();
    stats.time_marching_cubes = t0.elapsed().as_secs_f64();
    stats.time_total = t_total.elapsed().as_secs_f64();

    (Mesh { vertices: verts, triangles: unit_tris }, stats)
}
