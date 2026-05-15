use clap::Parser;

use poisson_recon::ply;
use poisson_recon::reconstructor::{self, ReconstructParams};

#[derive(Parser)]
#[command(name = "poisson-recon", version = "0.1.0", about = "Screened Poisson Surface Reconstruction")]
struct Args {
    #[arg(short, long)] input: String,
    #[arg(short, long, default_value = "output.ply")] output: String,
    #[arg(short, long, default_value = "8")] depth: u32,
    #[arg(long, default_value = "1.5")] samples_per_node: f64,
    #[arg(long, default_value = "0.001")] cg_accuracy: f64,
    #[arg(long, default_value = "300")] cg_iters: usize,
    #[arg(long, default_value = "0.0")] point_weight: f64,
    #[arg(long, default_value = "64")] grid_resolution: usize,
    #[arg(long, default_value = "1.1")] scale: f64,
    #[arg(short, long)] verbose: bool,
}

fn main() {
    let args = Args::parse();
    let points = match ply::read_oriented_points(&args.input) {
        Ok(pts) => pts,
        Err(e) => { eprintln!("Error reading input: {}", e); std::process::exit(1); }
    };
    if points.is_empty() { eprintln!("No points found."); std::process::exit(1); }
    if args.verbose { eprintln!("Read {} oriented points", points.len()); }

    let params = ReconstructParams {
        depth: args.depth, samples_per_node: args.samples_per_node,
        scale: args.scale, cg_accuracy: args.cg_accuracy, cg_iters: args.cg_iters,
        point_weight: args.point_weight, grid_resolution: args.grid_resolution,
        verbose: args.verbose,
    };

    let (mesh, stats) = reconstructor::reconstruct(&points, &params);

    if args.verbose {
        eprintln!(
            "Timing [s]: transform={:.3} tree={:.3} matrix={:.3} rhs={:.3} solve={:.3} eval={:.3} mc={:.3}",
            stats.time_transform, stats.time_tree_build, stats.time_matrix_assemble,
            stats.time_rhs_assemble, stats.time_solve, stats.time_evaluate, stats.time_marching_cubes
        );
        eprintln!("FEM nodes: {}, matrix NNZ: {}", stats.fem_nodes, stats.matrix_nnz);
        eprintln!("Mesh: {} verts, {} tris, iso={:.6}", stats.mesh_vertices, stats.mesh_triangles, stats.iso_value);
        eprintln!("Total wall time: {:.3} s", stats.time_total);
    }

    if let Err(e) = ply::write_mesh(&args.output, &mesh) {
        eprintln!("Error writing output: {}", e); std::process::exit(1);
    }
    if args.verbose { eprintln!("Wrote {}", args.output); }
}
