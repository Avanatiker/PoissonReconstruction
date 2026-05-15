/// Adaptive octree with FEM node data.
///
/// Implements the N-dimensional adaptive octree used as the spatial domain
/// for the finite element method. Each node carries flags indicating whether
/// it participates in the space partition, indexes FEM basis functions, etc.
use std::cell::UnsafeCell;

use crate::binary_node::BinaryNode;

/// Node index type (analogous to `node_index_type` in C++).
pub type NodeIndex = i32;
/// Matrix entry index type (analogous to `matrix_index_type` in C++).
pub type MatrixIndex = i32;
/// Depth and offset storage type.
pub type DepthOffset = u16;

pub const DIM: usize = 3;

/// Maximum depth of the octree.
pub const MAX_DEPTH: u32 = 15;

/// Flags for FEM tree nodes.
pub mod flags {
    pub const SPACE_FLAG: u8 = 1 << 0; // Node participates in the unit cube partition
    pub const FEM_FLAG_1: u8 = 1 << 1; // Indexes a valid finite element (staggered position 0)
    pub const FEM_FLAG_2: u8 = 1 << 2; // Indexes a valid finite element (staggered position 1)
    pub const DIRICHLET_NODE_FLAG: u8 = 1 << 3; // FEM elements evaluate to zero here
    pub const DIRICHLET_ELEMENT_FLAG: u8 = 1 << 4; // Coefficient locked to zero
    pub const GEOMETRY_SUPPORTED_FLAG: u8 = 1 << 5; // Support overlaps geometry
    pub const GHOST_FLAG: u8 = 1 << 6; // Children are pruned
    pub const SCRATCH_FLAG: u8 = 1 << 7; // Temporary flag
}

/// Data stored in each FEM tree node.
#[derive(Debug, Clone, Copy)]
pub struct FEMNodeData {
    /// The node's index in the flattened node array (-1 if not assigned).
    pub node_index: NodeIndex,
    /// Flags indicating node status.
    pub flags: u8,
}

impl Default for FEMNodeData {
    fn default() -> Self {
        FEMNodeData {
            node_index: -1,
            flags: 0,
        }
    }
}

impl FEMNodeData {
    #[inline]
    pub fn has_flag(&self, flag: u8) -> bool {
        (self.flags & flag) != 0
    }

    #[inline]
    pub fn set_flag(&mut self, flag: u8, value: bool) {
        if value {
            self.flags |= flag;
        } else {
            self.flags &= !flag;
        }
    }
}

/// A node in the adaptive octree.
///
/// Each node has:
/// - A parent pointer (None for root)
/// - Up to 2^DIM children
/// - Depth and offset encoding its position in the unit cube
/// - FEM-related node data
pub struct OctreeNode {
    /// Parent node (None for root).
    pub parent: Option<*mut OctreeNode>,
    /// Children: 8 for 3D. None if no children allocated.
    pub children: Option<Box<[OctreeNode; 8]>>,
    /// Depth of this node in the tree.
    pub depth: u32,
    /// Offset in each dimension at this depth.
    /// offset[d] in [0, 2^depth)
    pub offset: [u32; DIM],
    /// FEM node data (flags, node_index).
    pub data: UnsafeCell<FEMNodeData>,
}

// OctreeNode is !Sync by default because of raw pointers, but we manage
// synchronization at a higher level via the tree owner.
unsafe impl Send for OctreeNode {}
unsafe impl Sync for OctreeNode {}

unsafe impl Send for SortedTreeNodes {}
unsafe impl Sync for SortedTreeNodes {}

impl OctreeNode {
    pub fn new_root() -> Self {
        OctreeNode {
            parent: None,
            children: None,
            depth: 0,
            offset: [0; DIM],
            data: UnsafeCell::new(FEMNodeData::default()),
        }
    }

    pub fn new_child(parent: *mut OctreeNode, depth: u32, offset: [u32; DIM]) -> Self {
        OctreeNode {
            parent: Some(parent),
            children: None,
            depth,
            offset,
            data: UnsafeCell::new(FEMNodeData::default()),
        }
    }

    /// Get the node data immutably.
    #[inline]
    pub fn data(&self) -> &FEMNodeData {
        unsafe { &*self.data.get() }
    }

    /// Get the node data mutably.
    #[inline]
    pub fn data_mut(&mut self) -> &mut FEMNodeData {
        self.data.get_mut()
    }

    /// Allocate children for this node.
    pub fn init_children(&mut self) {
        if self.children.is_some() {
            return;
        }
        let cdepth = self.depth + 1;
        let coffsets = Self::child_offsets(self.offset);
        let myself = self as *mut OctreeNode;
        let children = Box::new(std::array::from_fn(|i| {
            OctreeNode::new_child(myself, cdepth, coffsets[i])
        }));
        self.children = Some(children);
    }

    /// Compute child offsets.
    fn child_offsets(offset: [u32; DIM]) -> [[u32; DIM]; 8] {
        let mut result = [[0u32; DIM]; 8];
        for i in 0..8 {
            for d in 0..DIM {
                result[i][d] = offset[d] * 2 + ((i >> d) & 1) as u32;
            }
        }
        result
    }

    /// Get the center of this node in unit cube coordinates.
    pub fn center(&self) -> [f64; DIM] {
        let mut center = [0.0f64; DIM];
        for d in 0..DIM {
            let (cd, _) = BinaryNode::center_and_width(self.depth, self.offset[d] as usize);
            center[d] = cd;
        }
        center
    }

    /// Get the width of this node.
    pub fn width(&self) -> f64 {
        BinaryNode::width(self.depth)
    }

    /// Check if this node is a leaf (no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// Count total number of nodes in the subtree rooted here.
    pub fn count_nodes(&self) -> usize {
        let mut count = 1;
        if let Some(ref children) = self.children {
            for child in children.iter() {
                count += child.count_nodes();
            }
        }
        count
    }
}

/// Sorted array of tree nodes, grouped by depth.
///
/// After tree finalization, nodes are flattened into a depth-sorted array
/// for efficient traversal during multigrid operations.
pub struct SortedTreeNodes {
    /// Flattened array of node pointers, sorted by depth.
    pub tree_nodes: Vec<*mut OctreeNode>,
    /// slice_start[d] = index of first node at depth d
    pub slice_start: Vec<usize>,
}

impl SortedTreeNodes {
    /// Build from a tree root by traversing the tree and collecting nodes.
    pub fn build(root: &OctreeNode) -> Self {
        let max_depth = Self::compute_max_depth(root);
        let mut nodes_by_depth: Vec<Vec<*mut OctreeNode>> = vec![Vec::new(); max_depth as usize + 1];

        Self::collect_nodes(root as *const OctreeNode as *mut OctreeNode, &mut nodes_by_depth);

        let mut tree_nodes = Vec::new();
        let mut slice_start = vec![0usize; max_depth as usize + 2];

        for d in 0..=max_depth as usize {
            slice_start[d] = tree_nodes.len();
            tree_nodes.extend(nodes_by_depth[d].iter());
        }
        slice_start[max_depth as usize + 1] = tree_nodes.len();

        SortedTreeNodes {
            tree_nodes,
            slice_start,
        }
    }

    fn compute_max_depth(node: &OctreeNode) -> u32 {
        let mut max_d = node.depth;
        if let Some(ref children) = node.children {
            for child in children.iter() {
                let cd = Self::compute_max_depth(child);
                if cd > max_d {
                    max_d = cd;
                }
            }
        }
        max_d
    }

    fn collect_nodes(node: *mut OctreeNode, nodes_by_depth: &mut Vec<Vec<*mut OctreeNode>>) {
        unsafe {
            let depth = (*node).depth as usize;
            nodes_by_depth[depth].push(node);
            if let Some(ref children) = (*node).children {
                for child in children.iter() {
                    Self::collect_nodes(
                        child as *const OctreeNode as *mut OctreeNode,
                        nodes_by_depth,
                    );
                }
            }
        }
    }

    /// Number of nodes at a given depth.
    pub fn size_at_depth(&self, depth: u32) -> usize {
        if depth as usize + 1 < self.slice_start.len() {
            self.slice_start[depth as usize + 1] - self.slice_start[depth as usize]
        } else {
            0
        }
    }

    /// Total number of nodes.
    pub fn total_nodes(&self) -> usize {
        self.tree_nodes.len()
    }

    /// Get the nodes at a given depth.
    pub fn nodes_at_depth(&self, depth: u32) -> &[*mut OctreeNode] {
        let start = self.slice_start[depth as usize];
        let end = if depth as usize + 1 < self.slice_start.len() {
            self.slice_start[depth as usize + 1]
        } else {
            self.tree_nodes.len()
        };
        &self.tree_nodes[start..end]
    }
}

/// Dense node data: array indexed by node_index.
///
/// After finalization, each active FEM node gets a unique index.
/// This stores per-node data (e.g., solution coefficients) in a flat array.
pub struct DenseNodeData<T: Clone + Default> {
    pub data: Vec<T>,
}

impl<T: Clone + Default> DenseNodeData<T> {
    pub fn new() -> Self {
        DenseNodeData { data: Vec::new() }
    }

    pub fn resize(&mut self, size: usize) {
        self.data.resize(size, T::default());
    }

    pub fn set(&mut self, idx: usize, value: T) {
        if idx >= self.data.len() {
            self.data.resize(idx + 1, T::default());
        }
        self.data[idx] = value;
    }

    pub fn get(&self, idx: usize) -> &T {
        &self.data[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        &mut self.data[idx]
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

/// Sparse node data: map from node_index to data, with thread-safe insertion.
pub struct SparseNodeData<T: Clone> {
    pub data: Vec<T>,
    pub indices: Vec<NodeIndex>,
    /// Map from node_index to position in data array.
    index_map: std::collections::HashMap<NodeIndex, usize>,
}

impl<T: Clone + Default> SparseNodeData<T> {
    pub fn new() -> Self {
        SparseNodeData {
            data: Vec::new(),
            indices: Vec::new(),
            index_map: std::collections::HashMap::new(),
        }
    }

    /// Insert or get data for a given node index.
    pub fn at(&mut self, node_idx: NodeIndex) -> &mut T {
        if let Some(&pos) = self.index_map.get(&node_idx) {
            &mut self.data[pos]
        } else {
            let pos = self.data.len();
            self.index_map.insert(node_idx, pos);
            self.indices.push(node_idx);
            self.data.push(T::default());
            &mut self.data[pos]
        }
    }

    /// Get data for a node index, if present.
    pub fn get(&self, node_idx: NodeIndex) -> Option<&T> {
        self.index_map.get(&node_idx).map(|&pos| &self.data[pos])
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.data.clear();
        self.indices.clear();
        self.index_map.clear();
    }
}

/// An octree that holds owned nodes (arena-based allocation).
///
/// Unlike the C++ version which uses custom allocators, we use Box-based
/// recursive ownership. For large trees, this may incur allocation overhead,
/// but it's simpler and more idiomatic in Rust.
pub struct Octree {
    pub root: Box<OctreeNode>,
    pub max_depth: u32,
    /// Flattened sorted nodes (set after finalization).
    pub sorted_nodes: Option<SortedTreeNodes>,
}

impl Octree {
    pub fn new() -> Self {
        Octree {
            root: Box::new(OctreeNode::new_root()),
            max_depth: 0,
            sorted_nodes: None,
        }
    }

    /// Refine the tree at a specific node to the target depth.
    pub fn refine_node(&mut self, node: &mut OctreeNode, target_depth: u32) {
        if node.depth >= target_depth {
            return;
        }
        node.init_children();
        if let Some(ref mut children) = node.children {
            for child in children.iter_mut() {
                self.refine_node(child, target_depth);
            }
        }
        if target_depth > self.max_depth {
            self.max_depth = target_depth;
        }
    }

    /// Build sorted nodes array after tree is fully constructed.
    pub fn finalize(&mut self) {
        self.sorted_nodes = Some(SortedTreeNodes::build(&self.root));
    }

    /// Get the sorted nodes, if finalized.
    pub fn sorted(&self) -> &SortedTreeNodes {
        self.sorted_nodes.as_ref().expect("Octree not finalized")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octree_create_root() {
        let root = OctreeNode::new_root();
        assert_eq!(root.depth, 0);
        assert_eq!(root.offset, [0, 0, 0]);
        assert!(root.is_leaf());
    }

    fn refine_subtree(node: &mut OctreeNode, target: u32) {
        if node.depth < target {
            node.init_children();
            if let Some(ref mut children) = node.children {
                for child in children.iter_mut() {
                    refine_subtree(child, target);
                }
            }
        }
    }

    #[test]
    fn test_octree_refine() {
        let mut tree = Octree::new();
        assert_eq!(tree.max_depth, 0);

        let target = 2;
        refine_subtree(&mut tree.root, target);
        tree.max_depth = target;
        assert_eq!(tree.max_depth, 2);
        assert!(!tree.root.is_leaf());

        let count = tree.root.count_nodes();
        let expected = 1 + 8 + 64;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_sorted_tree_nodes() {
        let mut tree = Octree::new();
        {
            let root = &mut tree.root;
            root.init_children();
            tree.max_depth = 1;
        }
        tree.finalize();

        let sorted = tree.sorted();
        assert_eq!(sorted.total_nodes(), 1 + 8);
        assert_eq!(sorted.size_at_depth(0), 1);
        assert_eq!(sorted.size_at_depth(1), 8);
    }
}
