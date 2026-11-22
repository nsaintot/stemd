//! Sorted, disjoint sample ranges.

use std::ops::Range;

/// Sorted, disjoint, half-open sample ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Spans(Vec<Range<usize>>);

impl Spans {
    /// Add `range`, merging it with any it touches.
    pub fn insert(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let (mut start, mut end) = (range.start, range.end);
        self.0.retain(|r| {
            if r.end < start || r.start > end {
                return true;
            }
            start = start.min(r.start);
            end = end.max(r.end);
            false
        });
        let at = self.0.partition_point(|r| r.start < start);
        self.0.insert(at, start..end);
    }

    /// True when one span holds all of `range`.
    pub fn covers(&self, range: &Range<usize>) -> bool {
        range.is_empty()
            || self
                .span_of(range.start)
                .is_some_and(|r| r.end >= range.end)
    }

    /// The span holding sample `i`.
    pub fn span_of(&self, i: usize) -> Option<&Range<usize>> {
        let at = self.0.partition_point(|r| r.end <= i);
        self.0.get(at).filter(|r| r.start <= i)
    }

    /// The parts of `range` already in a span.
    pub fn overlaps(&self, range: &Range<usize>) -> Vec<Range<usize>> {
        self.0
            .iter()
            .map(|r| r.start.max(range.start)..r.end.min(range.end))
            .filter(|r| !r.is_empty())
            .collect()
    }

    pub fn as_pairs(&self) -> Vec<[usize; 2]> {
        self.0.iter().map(|r| [r.start, r.end]).collect()
    }

    /// The number of samples in all spans together.
    pub fn total(&self) -> usize {
        self.0.iter().map(|r| r.end - r.start).sum()
    }
}
