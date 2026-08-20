//! What MLX is holding, and how to make it let go.
//!
//! MLX keeps freed buffers in an allocator cache instead of returning them to
//! the system. That is the right trade while work is arriving, since a track is
//! thousands of allocations of a handful of shapes, but the cache limit defaults
//! to the device's recommended working set, so an idle server goes on holding
//! most of the machine. One here sat on 14 GB, under half a gigabyte of which
//! was model weights, and pushed the machine 16.7 GB into swap.
//!
//! [`release`] is meant for an idle timer, not for between tracks, and it is
//! deliberately not a cache cap. A cap is worse than both: MLX evicts inside a
//! separation and re-allocates what it just gave up, and it still holds the
//! whole cap for ever. Releasing on idle costs nothing measurable, 38.29 s
//! against 38.31 s alternated with an unmodified binary.

/// Bytes MLX is holding for arrays that still exist.
pub fn active() -> usize {
    let mut bytes = 0;
    // SAFETY: writes one `size_t` through a pointer to a live local.
    unsafe { mlx_sys::mlx_get_active_memory(&mut bytes) };
    bytes
}

/// Bytes MLX is holding for arrays that no longer do. This is the number that
/// reached 13 GB.
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
