/// 1D binary tree indexing — maps between (depth, offset) and global indices.
///
/// This is the foundation for the N-dimensional octree indexing used by the FEMTree.
/// Centers and corners are indexed separately using cumulative prefix sums.
pub struct BinaryNode;

impl BinaryNode {
    /// Number of center positions at a given depth: 2^depth
    #[inline]
    pub fn center_count(depth: u32) -> usize {
        1usize << depth
    }

    /// Number of corner positions at a given depth: 2^depth + 1
    #[inline]
    pub fn corner_count(depth: u32) -> usize {
        (1usize << depth) + 1
    }

    /// Cumulative centers up to max_depth: 2^(maxDepth+1) - 1
    #[inline]
    pub fn cumulative_center_count(max_depth: u32) -> usize {
        (1usize << (max_depth + 1)) - 1
    }

    /// Cumulative corners up to max_depth: 2^(maxDepth+1) + maxDepth
    #[inline]
    pub fn cumulative_corner_count(max_depth: u32) -> usize {
        (1usize << (max_depth + 1)) + max_depth as usize
    }

    /// Global index for a center at (depth, offset): (2^depth) + offset - 1
    #[inline]
    pub fn center_index(depth: u32, offset: usize) -> usize {
        ((1usize << depth) + offset).wrapping_sub(1)
    }

    /// Global index for a corner at (depth, offset): (2^depth) + offset - 1 + depth
    #[inline]
    pub fn corner_index(depth: u32, offset: usize) -> usize {
        ((1usize << depth) + offset).wrapping_sub(1) + depth as usize
    }

    /// Corner index at max_depth for a corner at (depth, offset, forward_corner)
    #[inline]
    pub fn corner_index_at_max_depth(max_depth: u32, depth: u32, offset: usize, forward_corner: i32) -> usize {
        ((offset as isize + forward_corner as isize) as usize) << (max_depth - depth)
    }

    /// Width of a node at the given depth: 1 / 2^depth
    #[inline]
    pub fn width(depth: u32) -> f64 {
        1.0 / ((1usize << depth) as f64)
    }

    /// Center coordinate and width at (depth, offset)
    #[inline]
    pub fn center_and_width(depth: u32, offset: usize) -> (f64, f64) {
        let width = 1.0 / ((1usize << depth) as f64);
        let center = (offset as f64 + 0.5) * width;
        (center, width)
    }

    /// Corner coordinate and width at (depth, offset)
    #[inline]
    pub fn corner_and_width(depth: u32, offset: usize) -> (f64, f64) {
        let width = 1.0 / ((1usize << depth) as f64);
        let corner = offset as f64 * width;
        (corner, width)
    }

    /// Recover depth and offset from a global center index
    pub fn center_depth_and_offset(idx: usize) -> (u32, usize) {
        let mut offset = idx;
        let mut depth: u32 = 0;
        while offset >= (1usize << depth) {
            offset -= 1usize << depth;
            depth += 1;
        }
        (depth, offset)
    }

    /// Recover depth and offset from a global corner index
    pub fn corner_depth_and_offset(idx: usize) -> (u32, usize) {
        let mut offset = idx;
        let mut depth: u32 = 0;
        while offset >= ((1usize << depth) + 1) {
            offset -= (1usize << depth) + 1;
            depth += 1;
        }
        (depth, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_node_indexing() {
        for depth in 0..5 {
            assert_eq!(BinaryNode::center_count(depth), 1 << depth);
            assert_eq!(BinaryNode::corner_count(depth), (1 << depth) + 1);
        }

        assert_eq!(BinaryNode::center_index(0, 0), 0);
        assert_eq!(BinaryNode::center_index(1, 0), 1);
        assert_eq!(BinaryNode::center_index(1, 1), 2);

        let (d, o) = BinaryNode::center_depth_and_offset(2);
        assert_eq!(d, 1);
        assert_eq!(o, 1);
    }

    #[test]
    fn test_cumulative_counts() {
        assert_eq!(
            BinaryNode::cumulative_center_count(2),
            (1 << 0) + (1 << 1) + (1 << 2)
        );
    }
}
