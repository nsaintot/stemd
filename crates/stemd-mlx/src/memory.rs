//! Reports the memory MLX holds, and releases its cache.
//!
//! MLX keeps freed buffers in an allocator cache and does not return them to the system. That suits
//! a stream of work, since a track is thousands of allocations of a handful of shapes, but the
//! cache limit defaults to the device's recommended working set, so an idle server can keep most of
//! the machine's memory, almost none of it model weights, and push the machine into swap.
//!
//! Call [`release`] from an idle timer; between tracks it would slow the next track. A cache cap is
//! worse than releasing between tracks or on idle: MLX evicts inside a separation and re-allocates
//! what it just gave up, and it still holds the whole cap forever. Releasing on idle has no
//! measurable cost on the next separation.

/// The bytes MLX holds for arrays that still exist.
pub fn active() -> usize {
    let mut bytes = 0;
    // SAFETY: writes one `size_t` through a pointer to a live local.
    unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) };
    bytes
}

/// The bytes MLX holds for arrays that no longer exist.
pub fn cached() -> usize {
    let mut bytes = 0;
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_get_cache_memory(&mut bytes) };
    bytes
}

/// The most MLX has held at once since the process started.
pub fn peak() -> usize {
    let mut bytes = 0;
    // SAFETY: as above.
    unsafe { mlx_sys::mlx_get_peak_memory(&mut bytes) };
    bytes
}

/// Give back everything MLX is holding for arrays that are gone, and report how
/// much that was. Nothing live is touched, so a loaded model stays loaded.
pub fn release() -> usize {
    let before = cached();
    // SAFETY: frees only buffers no array refers to; MLX owns the bookkeeping.
    unsafe { mlx_sys::mlx_clear_cache() };
    before.saturating_sub(cached())
}
