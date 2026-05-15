use std::collections::HashMap;
use crate::geometry::{Point3, Vec3};
use crate::octree::{self, OctreeNode};
use crate::sparse::{MatrixEntry, SparseMatrix};
use crate::solvers;

pub const DIM: usize = 3;

#[derive(Debug, Clone)]
pub struct OrientedPoint { pub position: Point3, pub normal: Vec3 }

pub struct FEMTree {
    pub max_depth: u32,
    pub fem_node_count: usize,
    pub octree: octree::Octree,
    pub offset_to_idx: HashMap<[u32; DIM], usize>,
    pub idx_to_offset: Vec<[u32; DIM]>,
    pub solution: Vec<f64>,
    pub normal_field: Vec<Vec3>,
}

/// Standard marching cubes triangle table (256 entries, 16 i8 each).
/// Generated from the mcubes crate's TRI_TABLE (which mirrors the canonical Paul Bourke table).
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1],
    [3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1],
    [3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1],
    [3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1],
    [9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1],
    [9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1,-1],
    [8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1],
    [3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1],
    [1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1],
    [4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1,-1],
    [9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1,-1],
    [0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1,-1],
    [2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1,-1],
    [10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1,-1],
    [5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1,-1],
    [0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1,-1],
    [1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1,-1],
    [2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1],
    [7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1,-1],
    [9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1,-1],
    [11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1,-1],
    [5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],
    [11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1,-1],
    [9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1,-1],
    [2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1,-1],
    [6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1,-1],
    [6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1,-1],
    [6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1,-1],
    [8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],
    [3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1,-1],
    [0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],
    [8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],
    [6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1,-1],
    [10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1,-1],
    [10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1,-1],
    [1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1,-1],
    [0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1],
    [10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1,-1],
    [0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1,-1],
    [6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1,-1],
    [8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1,-1],
    [6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1,-1],
    [10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1,-1],
    [10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1,-1],
    [1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],
    [7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1,-1],
    [7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1],
    [11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1,-1],
    [8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1],
    [0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1,-1],
    [7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1,-1],
    [7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1,-1],
    [2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1,-1],
    [1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1,-1],
    [10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1,-1],
    [10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1,-1],
    [0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1,-1],
    [7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1,-1],
    [6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1,-1],
    [9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1,-1],
    [6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1,-1],
    [4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1,-1],
    [10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1],
    [8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1],
    [0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1,-1],
    [1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1,-1],
    [8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1,-1],
    [10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1,-1],
    [4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1],
    [10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1,-1],
    [5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1,-1],
    [9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1,-1],
    [6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1,-1],
    [7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1,-1],
    [3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1],
    [7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1,-1],
    [9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1,-1],
    [3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1,-1],
    [6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1,-1],
    [6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1,-1],
    [3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1,-1],
    [6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1,-1],
    [9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],
    [1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1,-1],
    [0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1,-1],
    [5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1,-1],
    [10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1,-1],
    [11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1,-1],
    [7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1],
    [8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1,-1],
    [9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1,-1],
    [9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1],
    [5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1,-1],
    [5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1,-1],
    [8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1,-1],
    [0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1,-1],
    [9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1,-1],
    [0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1,-1],
    [11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1,-1],
    [4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1,-1],
    [0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1,-1],
    [3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1,-1],
    [3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1,-1],
    [0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1,-1],
    [9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1,-1],
    [1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
];

/// Standard marching cubes edge table (256 entries).
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

impl FEMTree {
    pub fn new(max_depth: u32) -> Self {
        FEMTree { max_depth, fem_node_count: 0, octree: octree::Octree::new(),
            offset_to_idx: HashMap::new(), idx_to_offset: Vec::new(),
            solution: Vec::new(), normal_field: Vec::new() }
    }

    pub fn initialize_from_points(&mut self, points: &[OrientedPoint], threshold: f64) {
        let res = 1usize << self.max_depth; let h = 1.0 / res as f64;
        let mut cell_counts: HashMap<[u32; DIM], f64> = HashMap::new();
        for pt in points {
            let mut o = [0u32; DIM]; let mut ok = true;
            for d in 0..DIM { let c = (pt.position[d]/h) as isize; if c<0||c>=res as isize{ok=false;break;} o[d]=c as u32; }
            if ok { *cell_counts.entry(o).or_insert(0.0) += 1.0; }
        }
        let maxd = self.max_depth as usize;
        let mut dc: Vec<HashMap<[u32;DIM],f64>> = (0..=maxd).map(|_| HashMap::new()).collect();
        dc[maxd] = cell_counts;
        for d in (0..maxd).rev() {
            let mut m = HashMap::new();
            for (&[cx,cy,cz],&c) in &dc[d+1] { *m.entry([cx>>1,cy>>1,cz>>1]).or_insert(0.0) += c; }
            dc[d] = m;
        }
        fn refine(node: &mut OctreeNode, d: u32, off: [u32; DIM], mx: u32, dc: &[HashMap<[u32;DIM],f64>], th: f64) {
            if d>=mx {return;}
            let cnt = dc[d as usize].get(&off).copied().unwrap_or(0.0);
            if cnt<=th {return;}
            node.init_children();
            if let Some(ref mut kids) = node.children { for i in 0..8u32 {
                let o=[off[0]*2+(i&1),off[1]*2+((i>>1)&1),off[2]*2+((i>>2)&1)];
                refine(&mut kids[i as usize],d+1,o,mx,dc,th);
            }}
        }
        refine(&mut self.octree.root,0,[0;DIM],self.max_depth,&dc,threshold);
        if self.max_depth > self.octree.max_depth { self.octree.max_depth = self.max_depth; }
    }

    pub fn finalize(&mut self) {
        self.octree.finalize();
        let sorted = self.octree.sorted_nodes.as_ref().expect("not finalized");
        self.offset_to_idx.clear(); self.idx_to_offset.clear();
        let mut idx = 0;
        for &np in &sorted.tree_nodes {
            unsafe {
                let n = &mut *np;
                if n.is_leaf() && n.depth == self.max_depth {
                    n.data_mut().set_flag(octree::flags::SPACE_FLAG, true);
                    n.data_mut().set_flag(octree::flags::FEM_FLAG_1, true);
                    n.data_mut().node_index = idx as octree::NodeIndex;
                    self.offset_to_idx.insert(n.offset, idx);
                    self.idx_to_offset.push(n.offset); idx += 1;
                }
            }
        }
        self.fem_node_count = idx; self.solution.resize(idx, 0.0);
    }

    pub fn splat_normal_field(&mut self, points: &[OrientedPoint]) {
        let n = self.fem_node_count; self.normal_field.resize(n, Vec3::ZERO);
        let res = 1usize << self.max_depth; let h = 1.0 / res as f64;
        for pt in points {
            let cx=(pt.position.x/h) as isize; let cy=(pt.position.y/h) as isize; let cz=(pt.position.z/h) as isize;
            let tx=pt.position.x/h-cx as f64; let ty=pt.position.y/h-cy as f64; let tz=pt.position.z/h-cz as f64;
            let bx0=1.0-tx; let bx1=tx; let by0=1.0-ty; let by1=ty; let bz0=1.0-tz; let bz1=tz;
            let nx=-pt.normal.x; let ny=-pt.normal.y; let nz=-pt.normal.z;
            for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                let ox=(cx+dx_i) as u32; let oy=(cy+dy_i) as u32; let oz=(cz+dz_i) as u32;
                if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]) {
                    let bx=if dx_i==0{bx0}else{bx1}; let by=if dy_i==0{by0}else{by1}; let bz=if dz_i==0{bz0}else{bz1};
                    let w=bx*by*bz;
                    if idx<n { self.normal_field[idx].x+=nx*w; self.normal_field[idx].y+=ny*w; self.normal_field[idx].z+=nz*w; }
                }
            }}}
        }
    }

    fn m1(d: isize)->f64{match d.abs(){0=>2./3.,1=>1./6.,_=>0.}}
    fn k1(d: isize)->f64{match d.abs(){0=>2.,1=>-1.,_=>0.}}
    fn dm(d: isize)->f64{match d{0=>0.,1=>-0.5,-1=>0.5,_=>0.}}

    pub fn assemble_system_matrix(&self, points: &[OrientedPoint], _d: u32, pw: f64) -> SparseMatrix<f64> {
        let n=self.fem_node_count; let res=1usize<<self.max_depth; let h=1.0/res as f64;
        let mut re:Vec<Vec<(usize,f64)>>=vec![Vec::new();n];
        for i in 0..n{let[ix,iy,iz]=self.idx_to_offset[i];
            for dx in -1isize..=1isize{for dy in -1isize..=1isize{for dz in -1isize..=1isize{
                let jx=ix as isize+dx;let jy=iy as isize+dy;let jz=iz as isize+dz;
                if jx<0||jx>=res as isize||jy<0||jy>=res as isize||jz<0||jz>=res as isize{continue;}
                if let Some(&j)=self.offset_to_idx.get(&[jx as u32,jy as u32,jz as u32]){
                    let v=Self::k1(dx)*Self::m1(dy)*Self::m1(dz)+Self::m1(dx)*Self::k1(dy)*Self::m1(dz)+Self::m1(dx)*Self::m1(dy)*Self::k1(dz);
                    let val=v*h; if val!=0.0{re[i].push((j,val));}
                }
            }}}
        }
        if pw>0.0 { let wh3=pw*h.powi(3);
            for pt in points {
                let cx=(pt.position.x/h)as isize;let cy=(pt.position.y/h)as isize;let cz=(pt.position.z/h)as isize;
                let tx=pt.position.x/h-cx as f64;let ty=pt.position.y/h-cy as f64;let tz=pt.position.z/h-cz as f64;
                let bx0=1.-tx;let bx1=tx;let by0=1.-ty;let by1=ty;let bz0=1.-tz;let bz1=tz;
                // Collect basis function values and indices for this point (up to 8 nodes)
                let mut basis: Vec<(usize, f64)> = Vec::with_capacity(8);
                for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                    let ox=(cx+dx_i)as u32;let oy=(cy+dy_i)as u32;let oz=(cz+dz_i)as u32;
                    if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){
                        let bx=if dx_i==0{bx0}else{bx1};let by=if dy_i==0{by0}else{by1};let bz=if dz_i==0{bz0}else{bz1};
                        basis.push((idx, bx*by*bz));
                    }
                }}}
                // Add full sparse mass block: M[i,j] += w * phi_i * phi_j for all i,j in support
                for &(i_idx, phi_i) in &basis {
                    for &(j_idx, phi_j) in &basis {
                        re[i_idx].push((j_idx, wh3 * phi_i * phi_j));
                    }
                }
            }
        }
        for row in &mut re { row.sort_by_key(|(j,_)|*j);let mut j=0;let mut u=0;
            while j<row.len(){let col=row[j].0;let mut s=0.;
                while j<row.len()&&row[j].0==col{s+=row[j].1;j+=1;}row[u]=(col,s);u+=1;}row.truncate(u);}
        let nnz:usize=re.iter().map(|r|r.len()).sum();
        let mut mat=SparseMatrix::with_capacity(n,nnz);
        for i in 0..n{mat.set_row_size(i,re[i].len());}mat.finalize_structure();
        for i in 0..n{let row=mat.row_mut(i);for(k,&(col,val))in re[i].iter().enumerate(){row[k]=MatrixEntry::new(col,val);}}
        mat
    }

    pub fn assemble_rhs_from_field(&self, points: &[OrientedPoint], _d: u32, pw: f64) -> Vec<f64> {
        let n=self.fem_node_count;let res=1usize<<self.max_depth;let h=1.0/res as f64;let mut rhs=vec![0.0;n];
        for i in 0..n{let[ix,iy,iz]=self.idx_to_offset[i];let mut s=0.;
            for dx in -1isize..=1isize{for dy in -1isize..=1isize{for dz in -1isize..=1isize{
                let jx=ix as isize+dx;let jy=iy as isize+dy;let jz=iz as isize+dz;
                if jx<0||jx>=res as isize||jy<0||jy>=res as isize||jz<0||jz>=res as isize{continue;}
                if let Some(&j)=self.offset_to_idx.get(&[jx as u32,jy as u32,jz as u32]){
                    if j<self.normal_field.len(){
                        let v=self.normal_field[j];
                        s-=v.x*Self::dm(dx)*Self::m1(dy)*Self::m1(dz)+v.y*Self::m1(dx)*Self::dm(dy)*Self::m1(dz)+v.z*Self::m1(dx)*Self::m1(dy)*Self::dm(dz);
                    }
                }
            }}}rhs[i]=s;}
        if pw>0.0{let wh3=pw*h.powi(3);let t=0.5;
            for pt in points{
                let cx=(pt.position.x/h)as isize;let cy=(pt.position.y/h)as isize;let cz=(pt.position.z/h)as isize;
                let tx=pt.position.x/h-cx as f64;let ty=pt.position.y/h-cy as f64;let tz=pt.position.z/h-cz as f64;
                let bx0=1.-tx;let bx1=tx;let by0=1.-ty;let by1=ty;let bz0=1.-tz;let bz1=tz;
                for dz_i in 0..=1isize{for dy_i in 0..=1isize{for dx_i in 0..=1isize{
                    let ox=(cx+dx_i)as u32;let oy=(cy+dy_i)as u32;let oz=(cz+dz_i)as u32;
                    if ox>=res as u32||oy>=res as u32||oz>=res as u32{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){
                        let bx=if dx_i==0{bx0}else{bx1};let by=if dy_i==0{by0}else{by1};let bz=if dz_i==0{bz0}else{bz1};
                        rhs[idx]+=wh3*t*bx*by*bz;
                    }
                }}}
            }
        }
        rhs
    }

    pub fn solve_cascadic(&mut self, a: &SparseMatrix<f64>, b: &[f64], gs: usize, cg: usize, eps: f64) {
        let n=a.num_rows();self.solution.resize(n,0.);self.solution.fill(0.);
        for _ in 0..gs{solvers::gauss_seidel_sweep(a,b,&mut self.solution);}
        solvers::solve_cg(a,b,&mut self.solution,cg,eps);
    }

    pub fn evaluate(&self, p: &Point3, _d: u32) -> f64 {
        let res=1usize<<self.max_depth;let h=1.0/res as f64;
        let sx=p.x/h;let sy=p.y/h;let sz=p.z/h;
        #[inline]fn co(s:f64,r:usize)->(u32,u32){let f=s.floor()as isize;let r=r as isize;((f-1).clamp(0,r-1)as u32,f.clamp(0,r-1)as u32)}
        #[inline]fn bv(s:f64,o:u32)->f64{let d=s-o as f64;if d<0.||d>2.{0.}else if d<=1.{d}else{2.-d}}
        let(ox0,ox1)=co(sx,res);let(oy0,oy1)=co(sy,res);let(oz0,oz1)=co(sz,res);
        let bx0=bv(sx,ox0);let bx1=bv(sx,ox1);let by0=bv(sy,oy0);let by1=bv(sy,oy1);let bz0=bv(sz,oz0);let bz1=bv(sz,oz1);
        let mut v=0.;
        for&(ox,bx)in&[(ox0,bx0),(ox1,bx1)]{if bx==0.{continue;}
            for&(oy,by)in&[(oy0,by0),(oy1,by1)]{if by==0.{continue;}
                for&(oz,bz)in&[(oz0,bz0),(oz1,bz1)]{if bz==0.{continue;}
                    if let Some(&idx)=self.offset_to_idx.get(&[ox,oy,oz]){if idx<self.solution.len(){v+=self.solution[idx]*bx*by*bz;}}
                }}}
        v
    }

    /// Extract iso-surface by walking octree leaves with standard MC triTable.
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

            let mut cv = [0.0f64; 8]; let mut mci: usize = 0;
            for k in 0..8 {
                let c = co[k];
                let p = Point3::new(
                    (off[0] as u64 + c[0] as u64) as f64 * w,
                    (off[1] as u64 + c[1] as u64) as f64 * w,
                    (off[2] as u64 + c[2] as u64) as f64 * w,
                );
                cv[k] = self.evaluate(&p, self.max_depth);
                if cv[k] < iso { mci |= 1 << k; }
            }

            let em = EDGE_TABLE[mci] as u16;
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

                let h = (pos[0]*1e7) as u64 ^ (pos[1]*1e7) as u64 ^ (pos[2]*1e7) as u64;
                if let Some(&ex) = edge_hash.get(&h) {
                    evi[e] = ex;
                } else {
                    let idx = verts.len(); edge_hash.insert(h, idx); verts.push(pos); evi[e] = idx;
                }
            }

            let tt = &TRI_TABLE[mci];
            for i in (0..16).step_by(3) {
                if tt[i] < 0 { break; }
                tris.push([evi[tt[i] as usize], evi[tt[i+1] as usize], evi[tt[i+2] as usize]]);
            }
        }

        (verts, tris)
    }
}
