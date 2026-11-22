//! The `/v2/jobs` handlers.

use std::sync::Arc;

use anyhow::Context as _;
use axum::Json;
use axum::body::Body;
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::http::{self, StatusCode};
use axum::response::{IntoResponse, Response};
use stemd_core::{Audio, Stage, resample};
use tokio_util::io::ReaderStream;

use crate::cache::Cache;
use crate::jobs::Entry;
use crate::output::Output;
use crate::queue::QueuedWork;
use crate::state::{AppState, Refused};

use super::error::ApiError;
use super::params::{CreateParams, Submission};

/// Decode the upload to planar float at the model's rate, off the async runtime, with the general
/// resampler.
pub(super) async fn decode_upload(body: Bytes, submission: Submission) -> Result<Audio, ApiError> {
    let model_rate = submission.model_rate;
    let decoded = tokio::task::spawn_blocking(move || -> anyhow::Result<Audio> {
        let audio = Audio::from_interleaved(
            &body,
            submission.in_format,
            submission.channels,
            submission.sample_rate,
        )?;
        if audio.sample_rate == model_rate {
            return Ok(audio);
        }
        let began = std::time::Instant::now();
        let converted = resample::to_rate(&audio, model_rate)
            .with_context(|| format!("converting the upload from {} Hz", audio.sample_rate))?;
        tracing::info!(
            "upload at {} Hz, converted to the model's {model_rate} Hz in {:.2?}",
            audio.sample_rate,
            began.elapsed()
        );
        Ok(converted)
    })
    .await
    .map_err(|e| ApiError::internal(format!("decoding the upload: {e}")))?;

    decoded.map_err(|err| ApiError::bad_request(format!("{err:#}")))
}

/// The track id of `mix`, off the async runtime.
pub(super) async fn identify(mix: Audio) -> Result<(Audio, String), ApiError> {
    tokio::task::spawn_blocking(move || {
        let id = crate::ident::track_id(&mix);
        (mix, id)
    })
    .await
    .map_err(|e| ApiError::internal(format!("hashing the upload: {e}")))
}

/// [`crate::held::cached_entry`], off the async runtime.
pub(super) async fn cached_result(
    cache: &Arc<Cache>,
    cache_id: String,
    output: Output,
    id: String,
) -> Option<Arc<Entry>> {
    let cache = Arc::clone(cache);
    tokio::task::spawn_blocking(move || crate::held::cached_entry(&cache, &cache_id, &output, &id))
        .await
        .ok()
        .flatten()
}

/// A job as the client sees it now. The queue position is resolved at read time, because positions
/// shift as the worker takes jobs.
pub(crate) fn live_view(state: &AppState, job: &crate::jobs::Job) -> crate::jobs::JobView {
    let mut view = job.view();
    if view.progress.stage == Stage::Queued
        && let Some(progress) = state.queue.queued_progress_for(&job.id)
    {
        view.progress = progress;
    }
    view
}

pub(super) async fn create_job(
    State(state): State<Arc<AppState>>,
    Query(params): Query<CreateParams>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let (status, job) = submit(&state, &params, body).await?;
    Ok((status, Json(live_view(&state, &job))).into_response())
}

/// Start a job, join the one running for the same track, or answer from the cache.
pub(crate) async fn submit(
    state: &Arc<AppState>,
    params: &CreateParams,
    body: Bytes,
) -> Result<(StatusCode, Arc<crate::jobs::Job>), ApiError> {
    if let Some(key) = &params.file {
        crate::api::stem::check_file_key(key)?;
    }
    let submission = Submission::parse(params, &body, state)?;
    let mix = decode_upload(body, submission.clone()).await?;
    let (mix, id) = identify(mix).await?;
    // Record the file key before the claim, so a lookup by that key finds the job while it runs.
    if let Some(key) = &params.file
        && let Err(err) = state.cache.record_file(key, &id)
    {
        tracing::warn!("recording file key {key}: {err:#}");
    }
    let cache_id = state.switcher.cache_id();
    let key = crate::jobs::job_key(&id, &submission.output, &cache_id);

    // Claim before the cache lookup: `claim` registers the job under the same lock that looks for
    // one, so simultaneous requests for the same track share a single separation.
    let job = match state.jobs.claim(&key, crate::jobs::MixShape::of(&mix)) {
        Err(existing) => {
            tracing::debug!(
                job = %existing.id,
                "joining a separation already under way for track {id}"
            );
            return Ok((StatusCode::ACCEPTED, existing));
        }
        Ok(job) => crate::jobs::Claimed::new(job),
    };

    if let Some(entry) = cached_result(
        &state.cache,
        cache_id,
        submission.output.clone(),
        id.clone(),
    )
    .await
    {
        let job = job.disarm();
        tracing::debug!(job = %job.id, "serving track {id} from the cache");
        job.complete(entry, true);
        return Ok((StatusCode::OK, job));
    }

    let job = job.disarm();
    enqueue(state, &job, mix, submission)?;
    Ok((StatusCode::ACCEPTED, job))
}

/// Hand the decoded mix to the worker, failing the job if the disk or the queue is full.
pub(super) fn enqueue(
    state: &AppState,
    job: &Arc<crate::jobs::Job>,
    mix: Audio,
    submission: Submission,
) -> Result<(), ApiError> {
    let duration = mix.duration_secs();
    let position = state
        .queue_claimed(QueuedWork {
            job: Arc::clone(job),
            mix,
            output: submission.output.clone(),
            cache_id: state.switcher.cache_id(),
            from_api: true,
        })
        .map_err(|refused| match refused {
            Refused::Disk(full) => {
                ApiError::new(StatusCode::INSUFFICIENT_STORAGE, full.to_string())
            }
            Refused::Full(full) => ApiError::new(StatusCode::TOO_MANY_REQUESTS, format!("{full}")),
        })?;

    tracing::info!(
        job = %job.id,
        "queued {duration:.1}s of audio at position {position} ({} ch @ {} Hz, {} in, \
         {} parts out at {} Hz)",
        submission.channels,
        submission.sample_rate,
        submission.in_format,
        submission.output.parts(),
        submission.output.rate,
    );
    Ok(())
}

pub(super) async fn get_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    Ok(Json(live_view(&state, &job)).into_response())
}

/// Add a waiter to a job, so another client's DELETE leaves it running.
pub(super) async fn join_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    job.join();
    Ok(Json(live_view(&state, &job)).into_response())
}

pub(crate) async fn delete_job(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;

    // Two clients can share one job, so only the last waiter's `DELETE` stops the work.
    let remaining = job.release();
    if remaining > 0 {
        tracing::debug!(job = %id, "released, {remaining} waiter(s) still want it");
        return Ok(StatusCode::NO_CONTENT.into_response());
    }

    // Cancel before removing: `cancel` matches on the running job's id. Removing first would widen
    // the gap in which a new submission of the same track cannot join.
    if state.stop_job(&id) {
        tracing::info!(job = %id, "cancelled");
    }
    Ok(StatusCode::NO_CONTENT.into_response())
}

pub(crate) async fn get_stem(
    State(state): State<Arc<AppState>>,
    Path((id, name)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let job = state.jobs.get(&id).ok_or_else(ApiError::not_found)?;
    let entry =
        job.entry.lock().clone().ok_or_else(|| {
            ApiError::new(StatusCode::CONFLICT, "job has not finished separating")
        })?;
    let path = entry
        .stem(&name)
        .ok_or_else(ApiError::not_found)?
        .path
        .clone();

    let file = tokio::fs::File::open(&path).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            // The stem was deleted from the cache between the lookup and the open.
            ApiError::new(
                StatusCode::GONE,
                "these stems have left the cache; resubmit the track",
            )
        } else {
            ApiError::internal(format!("opening {}: {e}", path.display()))
        }
    })?;
    let len = file
        .metadata()
        .await
        .map(|m| m.len())
        .map_err(|e| ApiError::internal(format!("stat {}: {e}", path.display())))?;

    // Mark the stem fetched only once its bytes are readable, so a failed fetch keeps the stems
    // held.
    entry.mark_fetched(&name);

    Ok(Response::builder()
        .header(
            http::header::CONTENT_TYPE,
            entry.output.format.content_type(),
        )
        .header(http::header::CONTENT_LENGTH, len)
        .body(Body::from_stream(ReaderStream::new(file)))
        .expect("response builder inputs are valid"))
}
