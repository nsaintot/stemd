//! 16-bit quantisation and FLAC encoding shared by the server and the CLI.

use anyhow::Result;
use flacenc::error::Verified;

/// The multiplier that turns `[-1.0, 1.0]` into an `i16`, and the divisor that turns a 16-bit stem
/// back. It is one less than 32768, so `+1.0` lands on `i16::MAX` and does not wrap.
pub const FLOAT_TO_S16: f32 = 32767.0;

/// Quantise one sample to 16 bits, clamping first, since a separated stem can sit outside `[-1.0,
/// 1.0]`.
pub fn quantise_s16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * FLOAT_TO_S16).round() as i16
}

/// The bit depth of FLAC stems.
///
/// 16 bits suffice: the quantisation floor is already about 60 dB below the model's own error. FLAC
/// has no float encoding, so `f32le` has no FLAC form.
pub const FLAC_BITS: usize = 16;

/// A FLAC encoder configuration at the crate defaults, with `lpc_order` when given.
pub fn flac_config(lpc_order: Option<usize>) -> Result<Verified<flacenc::config::Encoder>> {
    use flacenc::error::Verify;

    let mut config = flacenc::config::Encoder::default();
    if let Some(order) = lpc_order {
        config.subframe_coding.qlpc.lpc_order = order;
    }
    config
        .into_verified()
        .map_err(|e| anyhow::anyhow!("flac encoder config: {e:?}"))
}

/// Encode planar float to FLAC at [`FLAC_BITS`], each sample scaled by `gain` and quantised by
/// [`quantise_s16`].
pub fn encode_flac(
    planar: &[Vec<f32>],
    sample_rate: u32,
    gain: f32,
    config: &Verified<flacenc::config::Encoder>,
) -> Result<Vec<u8>> {
    use flacenc::component::BitRepr;

    let channels = planar.len();
    anyhow::ensure!(channels > 0, "cannot encode zero channels to flac");

    let frames = planar[0].len();
    let mut interleaved = Vec::with_capacity(frames * channels);
    for i in 0..frames {
        for ch in planar {
            interleaved.push(i32::from(quantise_s16(ch[i] * gain)));
        }
    }

    let source = flacenc::source::MemSource::from_samples(
        &interleaved,
        channels,
        FLAC_BITS,
        sample_rate as usize,
    );
    let mut stream = flacenc::encode_with_fixed_block_size(config, source, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac encode: {e}"))?;

    // The encoder folds a trailing partial block into STREAMINFO's minimum, which
    // declares a variable-block-size stream whose frames are fixed. A decoder that
    // trusts it locates frames by sample number and stops early.
    stream
        .stream_info_mut()
        .set_block_sizes(config.block_size, config.block_size)
        .map_err(|e| anyhow::anyhow!("flac block sizes: {e:?}"))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| anyhow::anyhow!("flac serialise: {e}"))?;
    Ok(sink.into_inner())
}
