//! The tensors of a converted artefact, addressed by name.
//!
//! Every layer builds itself from a prefix of this map. A missing key is an error
//! naming the key, so a model assembled from the wrong artefact says which tensor
//! it wanted.
//!
//! Nothing is transposed here. The artefacts are converted for MLX, whose convolution weights are
//! already `[out, kernel, in]`. Weights taken from PyTorch directly would need a transpose, and the
//! per-layer shape assertions would catch its absence.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use mlx_rs::Array;

use crate::precision::Precision;

pub struct Weights {
    tensors: HashMap<String, Array>,
}

impl Weights {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let tensors = Array::load_safetensors(
            path.to_str()
                .with_context(|| format!("{} is not utf-8", path.display()))?,
        )
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
        Ok(Self { tensors })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// The same tensors at another precision.
    ///
    /// Casting the weights is enough to move the network to half precision: mlx promotes, so a half
    /// weight meeting a half activation stays half, and the shape checks are unaffected. It takes a
    /// [`Precision`] so callers need not name mlx.
    pub fn cast(&self, precision: Precision) -> Result<Self> {
        let dtype = precision.dtype();
        let tensors = self
            .tensors
            .iter()
            .map(|(name, tensor)| Ok((name.clone(), tensor.as_dtype(dtype)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self { tensors })
    }

    /// The number of models the artefact holds.
    ///
    /// `htdemucs` holds one; `htdemucs_ft` holds four fine-tuned checkpoints in a single file,
    /// under `model_0` to `model_3`. Only the tensor names tell them apart.
    pub fn models(&self) -> usize {
        (0..)
            .take_while(|n| {
                let prefix = format!("model_{n}.");
                self.tensors.keys().any(|k| k.starts_with(&prefix))
            })
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A view scoped to `prefix`, so a layer names its own tensors.
    pub fn at<'a>(&'a self, prefix: &str) -> Scope<'a> {
        Scope {
            weights: self,
            prefix: prefix.to_owned(),
        }
    }

    fn get(&self, key: &str) -> Result<Array> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no tensor named {key}"))
    }
}

pub struct Scope<'a> {
    weights: &'a Weights,
    prefix: String,
}

impl<'a> Scope<'a> {
    /// It borrows the weights directly, so scopes chain: `w.child("layers").child("0")` builds a
    /// path without keeping a temporary alive.
    pub fn child(&self, name: &str) -> Scope<'a> {
        Scope {
            weights: self.weights,
            prefix: format!("{}.{name}", self.prefix),
        }
    }

    pub fn get(&self, name: &str) -> Result<Array> {
        self.weights.get(&format!("{}.{name}", self.prefix))
    }

    /// A tensor, checked against the shape the layer expects.
    ///
    /// The check matters most for convolutions: a weight in the wrong layout has the right number
    /// of elements and the wrong meaning, and the model would run and silently separate nothing.
    pub fn get_shaped(&self, name: &str, shape: &[i32]) -> Result<Array> {
        let array = self.get(name)?;
        if array.shape() != shape {
            bail!(
                "{}.{name} is {:?}, expected {shape:?}",
                self.prefix,
                array.shape()
            );
        }
        Ok(array)
    }

    pub fn has(&self, name: &str) -> bool {
        self.weights
            .tensors
            .contains_key(&format!("{}.{name}", self.prefix))
    }
}
