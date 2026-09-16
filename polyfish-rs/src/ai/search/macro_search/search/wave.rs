// In-flight leaf waiting on batched eval result.
// `count` tracks repeated picks of the same (parent, edge) for correct backup.
pub(super) struct PendingLeaf {
    // parent node index
    parent: usize,
    // edge index
    edge: usize,
    // path of nodes visited so far
    path: Vec<(usize, usize)>,
    // offset into the feature vector
    feat_offset: usize,
    // count of how many times this leaf has been picked
    count: u32,
}

/// Outcome of one `descend_once` call.
pub(super) enum DescendOutcome {
    /// A real value is already known — `backup` runs immediately.
    Resolved(f32),
    /// Queued into the wave and resolved later by `resolve_wave`.
    Deferred,
}
