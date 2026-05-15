/// Marching Cubes iso-surface extraction.
///
/// Thin wrapper around the `mcubes` crate for standard marching cubes
/// on a regular grid.

/// Result of marching cubes extraction.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
}

/// Extract iso-surface using standard marching cubes on a regular grid.
///
/// * `values` - 3D array of size `nx * ny * nz` in x-major order (x varying fastest, then y, then z)
/// * `iso` - iso-value
/// * `origin` - world position of the first grid sample
/// * `delta` - cell size in each dimension
pub fn marching_cubes_regular(
    values: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    iso: f64,
    origin: [f64; 3],
    _delta: [f64; 3],
) -> Mesh {
    // Convert f64 values to f32 for mcubes
    let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

    let size = (
        (nx - 1) as f32,
        (ny - 1) as f32,
        (nz - 1) as f32,
    );
    let sampling_interval = (1.0f32, 1.0f32, 1.0f32);

    let offset = lin_alg::f32::Vec3::new(origin[0] as f32, origin[1] as f32, origin[2] as f32);

    let mc = mcubes::MarchingCubes::new(
        (nx, ny, nz),
        size,
        sampling_interval,
        offset,
        values_f32,
        iso as f32,
    )
    .expect("Failed to create MarchingCubes");

    let mesh = mc.generate(mcubes::MeshSide::Both);

    // mcubes vertex_pos = (grid_index_position) * (nx-1)
    // To convert to unit cube [0,1]: unit = vertex / ((nx-1) * (nx-1))
    let vertices: Vec<[f64; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            let fx = v.posit.x as f64 / ((nx - 1) * (nx - 1)) as f64 + origin[0];
            let fy = v.posit.y as f64 / ((ny - 1) * (ny - 1)) as f64 + origin[1];
            let fz = v.posit.z as f64 / ((nz - 1) * (nz - 1)) as f64 + origin[2];
            [fx, fy, fz]
        })
        .collect();

    let triangles: Vec<[usize; 3]> = mesh
        .indices
        .chunks(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect();

    Mesh { vertices, triangles }
}

/// Merge duplicate vertices within epsilon distance.
pub fn merge_vertices(mesh: &mut Mesh, epsilon: f64) {
    use std::collections::HashMap;

    if mesh.vertices.is_empty() {
        return;
    }

    let scale = 1.0 / epsilon;
    let mut vertex_map: HashMap<[i64; 3], usize> = HashMap::new();
    let mut new_vertices: Vec<[f64; 3]> = Vec::new();
    let mut remap: Vec<usize> = vec![0; mesh.vertices.len()];

    for (i, v) in mesh.vertices.iter().enumerate() {
        let key = [
            (v[0] * scale).round() as i64,
            (v[1] * scale).round() as i64,
            (v[2] * scale).round() as i64,
        ];
        if let Some(&idx) = vertex_map.get(&key) {
            remap[i] = idx;
        } else {
            let idx = new_vertices.len();
            vertex_map.insert(key, idx);
            new_vertices.push(*v);
            remap[i] = idx;
        }
    }

    for tri in &mut mesh.triangles {
        tri[0] = remap[tri[0]];
        tri[1] = remap[tri[1]];
        tri[2] = remap[tri[2]];
    }

    mesh.vertices = new_vertices;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marching_cubes_sphere() {
        let n = 10usize;
        let mut values = vec![0.0f64; n * n * n];
        let center = [4.5f64, 4.5, 4.5];
        let radius = 2.5f64;

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dx = x as f64 - center[0];
                    let dy = y as f64 - center[1];
                    let dz = z as f64 - center[2];
                    values[z * n * n + y * n + x] =
                        radius - (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
        }

        let mesh = marching_cubes_regular(&values, n, n, n, 0.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(mesh.vertices.len() > 0);
        assert!(mesh.triangles.len() > 0);
    }
}
