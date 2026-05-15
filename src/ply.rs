/// Minimal PLY file reader/writer for oriented point clouds and triangle meshes.

use crate::fem_tree::OrientedPoint;
use crate::geometry::{Point3, Vec3};
use crate::marching_cubes::Mesh;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

/// Read an oriented point cloud from a PLY file.
///
/// Expects PLY format with `property float x`, `property float y`, `property float z`,
/// and `property float nx`, `property float ny`, `property float nz`.
pub fn read_oriented_points(path: &str) -> Result<Vec<OrientedPoint>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open {}: {}", path, e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Read header
    let mut num_vertices = 0usize;
    let mut has_normals = false;
    let mut in_header = true;

    let mut header_lines = Vec::new();

    while in_header {
        let line = lines
            .next()
            .ok_or("Unexpected end of file")?
            .map_err(|e| format!("Read error: {}", e))?;
        header_lines.push(line.clone());

        if line.starts_with("end_header") {
            in_header = false;
        } else if line.starts_with("element vertex ") {
            num_vertices = line["element vertex ".len()..]
                .parse()
                .map_err(|_| "Invalid vertex count")?;
        } else if line.contains("nx") || line.contains("normal_x") {
            has_normals = true;
        }
    }

    if !has_normals {
        return Err("PLY file does not contain normal data (nx, ny, nz)".to_string());
    }

    let mut points = Vec::with_capacity(num_vertices);

    for line_result in lines {
        let line = line_result.map_err(|e| format!("Read error: {}", e))?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }

        let px: f64 = parts[0].parse().unwrap_or(0.0);
        let py: f64 = parts[1].parse().unwrap_or(0.0);
        let pz: f64 = parts[2].parse().unwrap_or(0.0);
        let nx: f64 = parts[3].parse().unwrap_or(0.0);
        let ny: f64 = parts[4].parse().unwrap_or(0.0);
        let nz: f64 = parts[5].parse().unwrap_or(0.0);

        points.push(OrientedPoint {
            position: Point3::new(px, py, pz),
            normal: Vec3::new(nx, ny, nz),
        });
    }

    Ok(points)
}

/// Write a triangle mesh to a PLY file.
pub fn write_mesh(path: &str, mesh: &Mesh) -> Result<(), String> {
    let file = File::create(path).map_err(|e| format!("Cannot create {}: {}", path, e))?;
    let mut writer = BufWriter::new(file);

    writeln!(writer, "ply").map_err(|e| e.to_string())?;
    writeln!(writer, "format ascii 1.0").map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "element vertex {}",
        mesh.vertices.len()
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "property float x").map_err(|e| e.to_string())?;
    writeln!(writer, "property float y").map_err(|e| e.to_string())?;
    writeln!(writer, "property float z").map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "element face {}",
        mesh.triangles.len()
    )
    .map_err(|e| e.to_string())?;
    writeln!(writer, "property list uchar int vertex_indices")
        .map_err(|e| e.to_string())?;
    writeln!(writer, "end_header").map_err(|e| e.to_string())?;

    for v in &mesh.vertices {
        writeln!(writer, "{} {} {}", v[0], v[1], v[2]).map_err(|e| e.to_string())?;
    }

    for tri in &mesh.triangles {
        writeln!(writer, "3 {} {} {}", tri[0], tri[1], tri[2]).map_err(|e| e.to_string())?;
    }

    Ok(())
}
