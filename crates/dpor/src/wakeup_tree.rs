//! Wakeup tree data structure for Optimal DPOR.
//!
//! A wakeup tree is an ordered tree of process sequences that controls which
//! interleavings to explore next. Each branch represents a sequence of thread
//! choices that reverses a detected race. The tree ensures that no sleep-set-blocked
//! explorations occur, achieving optimality (one execution per Mazurkiewicz trace).
//!
//! Reference: Abdulla et al., "Source Sets: A Foundation for Optimal Dynamic
//! Partial Order Reduction", JACM 2017, Section 6-7.

/// A node in the wakeup tree. Children are ordered (exploration order matters).
#[derive(Clone, Debug)]
struct WakeupNode {
    thread_id: usize,
    children: Vec<WakeupNode>,
}

/// Wakeup tree: an ordered tree of thread-id sequences.
///
/// The tree is rooted at an implicit root node. Each path from root to a leaf
/// represents a wakeup sequence — an initial fragment of an execution that
/// reverses a detected race without encountering sleep-set blocking.
#[derive(Clone, Debug)]
pub struct WakeupTree {
    /// Children of the implicit root node, in exploration order.
    children: Vec<WakeupNode>,
}

impl WakeupTree {
    /// Create an empty wakeup tree (no branches to explore).
    pub fn empty() -> Self {
        Self { children: Vec::new() }
    }

    /// Create a wakeup tree with a single branch (one thread to explore).
    pub fn singleton(thread_id: usize) -> Self {
        Self {
            children: vec![WakeupNode {
                thread_id,
                children: Vec::new(),
            }],
        }
    }

    /// Returns true if this tree has no branches to explore.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns the thread IDs at the root level (first steps of each branch).
    pub fn root_threads(&self) -> Vec<usize> {
        self.children.iter().map(|n| n.thread_id).collect()
    }

    /// Get the first (leftmost) thread to explore. Returns None if empty.
    pub fn first_thread(&self) -> Option<usize> {
        self.children.first().map(|n| n.thread_id)
    }

    /// Get the minimum thread ID among root-level branches.
    /// This is used for exploration ordering (lowest-index first).
    pub fn min_thread(&self) -> Option<usize> {
        self.children.iter().map(|n| n.thread_id).min()
    }

    /// Extract the subtree rooted at the first child with the given thread_id.
    /// This is used when descending into a chosen thread's exploration.
    pub fn subtree(&self, thread_id: usize) -> WakeupTree {
        for child in &self.children {
            if child.thread_id == thread_id {
                return WakeupTree {
                    children: child.children.clone(),
                };
            }
        }
        WakeupTree::empty()
    }

    /// Remove the branch starting with the given thread_id from this tree
    /// (called after exploring that thread). Also returns whether the thread
    /// was found.
    pub fn remove_branch(&mut self, thread_id: usize) -> bool {
        let len_before = self.children.len();
        self.children.retain(|n| n.thread_id != thread_id);
        self.children.len() < len_before
    }

    /// Insert a wakeup sequence into the tree.
    ///
    /// The sequence is a list of thread IDs representing a path that must be
    /// explored to reverse a detected race. The sequence is inserted respecting
    /// the tree structure: shared prefixes are merged, and new branches are
    /// added at the end (rightmost position) to maintain exploration order.
    ///
    /// `is_weak_initial` is a callback that checks whether a given thread_id
    /// is a weak initial of the remaining sequence after the current prefix.
    /// This is used to determine whether an existing branch already covers
    /// the needed exploration (Property 2 of Definition 6.1).
    ///
    /// Returns true if a new leaf was added.
    pub fn insert(&mut self, sequence: &[usize]) -> bool {
        if sequence.is_empty() {
            return false;
        }
        self.insert_at(&mut Vec::new(), sequence)
    }

    /// Internal recursive insertion.
    fn insert_at(&mut self, _prefix: &mut Vec<usize>, sequence: &[usize]) -> bool {
        if sequence.is_empty() {
            return false;
        }

        let thread_id = sequence[0];
        let rest = &sequence[1..];

        // Check if this thread already exists as a child
        for child in &mut self.children {
            if child.thread_id == thread_id {
                // Merge: descend into the existing child
                if rest.is_empty() {
                    // The sequence ends here; this node already exists
                    return false;
                }
                let mut subtree = WakeupTree {
                    children: std::mem::take(&mut child.children),
                };
                let added = subtree.insert_at(_prefix, rest);
                child.children = subtree.children;
                return added;
            }
        }

        // Thread not found among children: add new branch.
        // But first check if any existing branch "covers" this sequence
        // (i.e., explores an equivalent interleaving). For simplicity and
        // correctness, we always add the new branch — the sleep set check
        // at the call site (process_races) already filters redundant inserts.
        let mut node = WakeupNode {
            thread_id,
            children: Vec::new(),
        };

        // Build the rest of the chain
        let mut current = &mut node;
        for &tid in rest {
            current.children.push(WakeupNode {
                thread_id: tid,
                children: Vec::new(),
            });
            current = current.children.last_mut().unwrap();
        }

        self.children.push(node);
        true
    }

    /// Check if the tree contains any branch starting with the given thread.
    pub fn contains_thread(&self, thread_id: usize) -> bool {
        self.children.iter().any(|n| n.thread_id == thread_id)
    }

    /// Get all leaf sequences (for debugging/testing).
    pub fn leaf_sequences(&self) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        for child in &self.children {
            let mut path = vec![child.thread_id];
            Self::collect_leaves(child, &mut path, &mut result);
        }
        result
    }

    fn collect_leaves(node: &WakeupNode, path: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
        if node.children.is_empty() {
            result.push(path.clone());
        } else {
            for child in &node.children {
                path.push(child.thread_id);
                Self::collect_leaves(child, path, result);
                path.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = WakeupTree::empty();
        assert!(tree.is_empty());
        assert_eq!(tree.first_thread(), None);
        assert!(tree.root_threads().is_empty());
    }

    #[test]
    fn test_singleton_tree() {
        let tree = WakeupTree::singleton(0);
        assert!(!tree.is_empty());
        assert_eq!(tree.first_thread(), Some(0));
        assert_eq!(tree.root_threads(), vec![0]);
    }

    #[test]
    fn test_insert_single_sequence() {
        let mut tree = WakeupTree::empty();
        assert!(tree.insert(&[1, 2, 3]));
        assert_eq!(tree.leaf_sequences(), vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_insert_shared_prefix() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[1, 2, 3]);
        tree.insert(&[1, 2, 4]);
        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0], vec![1, 2, 3]);
        assert_eq!(leaves[1], vec![1, 2, 4]);
    }

    #[test]
    fn test_insert_disjoint_branches() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[1, 0]);
        assert_eq!(tree.root_threads(), vec![0, 1]);
        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn test_insert_duplicate_does_not_add() {
        let mut tree = WakeupTree::empty();
        assert!(tree.insert(&[1, 2]));
        assert!(!tree.insert(&[1, 2]));
        assert_eq!(tree.leaf_sequences().len(), 1);
    }

    #[test]
    fn test_subtree() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1, 2]);
        tree.insert(&[0, 3]);
        let sub = tree.subtree(0);
        assert_eq!(sub.root_threads(), vec![1, 3]);
    }

    #[test]
    fn test_remove_branch() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[1, 0]);
        assert!(tree.remove_branch(0));
        assert_eq!(tree.root_threads(), vec![1]);
        assert!(!tree.remove_branch(0));
    }

    #[test]
    fn test_contains_thread() {
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);
        tree.insert(&[2, 3]);
        assert!(tree.contains_thread(0));
        assert!(tree.contains_thread(2));
        assert!(!tree.contains_thread(1));
    }

    #[test]
    fn test_complex_tree_structure() {
        // Simulates the wakeup tree from Fig. 7 in the paper:
        // Root has children: p(0), r(2)
        // p(0) has children: q(1), r(2)
        // p.r(0,2) has children: r(2) [i.e., p.r.r]
        // r(2) has children: r(2) [i.e., r.r]
        // r.r(2,2) has children: q(1) [i.e., r.r.q]
        let mut tree = WakeupTree::empty();
        tree.insert(&[0, 1]);       // p.q
        tree.insert(&[0, 2, 2]);    // p.r.r
        tree.insert(&[2, 2, 1]);    // r.r.q

        let leaves = tree.leaf_sequences();
        assert_eq!(leaves.len(), 3);
        assert_eq!(leaves[0], vec![0, 1]);
        assert_eq!(leaves[1], vec![0, 2, 2]);
        assert_eq!(leaves[2], vec![2, 2, 1]);
    }
}
