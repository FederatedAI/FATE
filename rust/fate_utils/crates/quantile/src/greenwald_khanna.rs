//! Greenwald Khanna calculates epsilon-approximate quantiles.
//! If the desired quantile is phi, the epsilon-approximate
//! quantile is any element in the range of elements that rank
//! between `lbound((phi-epsilon) x N)` and `lbound((phi+epsilon) x N)`
//!
//! terminology from the paper:
//!
//!   * S: set of observations
//!   * n: number of observations in S
//!   * v[i]: observation i in S
//!   * r: rank of observation in S from 1 to n.
//!   * `r_min(v[i])`: lower bound on rank r of v[i]
//!   * `r_max(v[i])`: upper bound on rank r of v[i]
//!   * `g[i] = r_min(v[i]) - r_min(v[i - 1])`
//!   * `delta[i] = r_max(v[i]) - r_min(v[i])`
//!   * `t[i] = tuple(v[i], g[i], delta[i])`
//!   * phi: quantile as a real number in the range [0,1]
//!   * r: ubound(phi * n)
//!
//! identities:
//!
//! * `r_min(v[i]) = forall j<=i sum of g[j]`
//! * `r_max(v[i]) = ( forall j<=i sum of g[j] ) + delta[i]`
//! * g[i] + delta[i] - 1 is an upper bound on the total number of observations
//! * between v[i] and v[i-1]
//! * sum of g[i] = n
//!
//! results:
//!
//! * `max_i(g[i] + delta[i]) <= 2 * epsilon * n`
//! * a tuple is full if g[i] + delta[i] = floor(2 * epsilon * n)
//!
//! `@inproceedings{Greenwald:2001:SOC:375663.375670,
//!       author = {Greenwald, Michael and Khanna, Sanjeev},
//!       title = {Space-efficient Online Computation of Quantile Summaries},
//! booktitle = {Proceedings of the 2001 ACM SIGMOD International
//! Conference
//!                    on Management of Data},
//!       series = {SIGMOD '01},
//!       year = {2001},
//!       isbn = {1-58113-332-4},
//!       location = {Santa Barbara, California, USA},
//!       pages = {58--66},
//!       numpages = {9},
//!       url = {http://doi.acm.org/10.1145/375663.375670},
//!       doi = {10.1145/375663.375670},
//!       acmid = {375670},
//!       publisher = {ACM},
//!       address = {New York, NY, USA},
//!     }`
//!
//! # Examples
//!
//! ```
//! use quantile::greenwald_khanna::*;
//!
//! let epsilon = 0.01;
//!
//! let mut stream = Stream::new(epsilon);
//!
//! let n = 1001;
//! for i in 1..n {
//!     stream.insert(i);
//! }
//! let in_range = |phi: f64, value: u32| {
//!   let lower = ((phi - epsilon) * (n as f64)) as u32;
//!   let upper = ((phi + epsilon) * (n as f64)) as u32;
//!   (epsilon > phi || lower <= value) && value <= upper
//! };
//! assert!(in_range(0f64, *stream.quantile(0f64)));
//! assert!(in_range(0.1f64, *stream.quantile(0.1f64)));
//! assert!(in_range(0.2f64, *stream.quantile(0.2f64)));
//! assert!(in_range(0.3f64, *stream.quantile(0.3f64)));
//! assert!(in_range(0.4f64, *stream.quantile(0.4f64)));
//! assert!(in_range(1f64, *stream.quantile(1f64)));
//! ```

use serde::{Deserialize, Serialize};
use std::cmp;
/// Locates the proper position of v in a vector vs
/// such that when v is inserted at position i,
/// it is less then the element at i+1 if any,
/// and greater than or equal to the element at i-1 if any.
pub fn find_insert_pos<T>(vs: &[T], v: &T) -> usize
where
    T: Ord,
{
    if vs.len() <= 10 {
        return find_insert_pos_linear(vs, v);
    }

    let middle = vs.len() / 2;
    let pivot = &vs[middle];

    if v < pivot {
        find_insert_pos(&vs[0..middle], v)
    } else {
        middle + find_insert_pos(&vs[middle..], v)
    }
}

/// Locates the proper position of v in a vector vs
/// such that when v is inserted at position i,
/// it is less then the element at i+1 if any,
/// and greater than or equal to the element at i-1 if any.
/// Works by scanning the slice from start to end.
pub fn find_insert_pos_linear<T>(vs: &[T], v: &T) -> usize
where
    T: Ord,
{
    for (i, vi) in vs.iter().enumerate() {
        if v < vi {
            return i;
        }
    }

    vs.len()
}

/// 3-tuple of a value v[i], g[i] and delta[i].
#[derive(Eq, Ord, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tuple<T>
where
    T: Ord,
{
    /// v[i], an observation in the set of observations
    pub v: T,

    /// the difference between the rank lowerbounds of t[i] and t[i-1]
    /// g = r_min(v[i]) - r_min(v[i - 1])
    pub g: usize,

    /// the difference betweeh the rank upper and lower bounds for this tuple
    pub delta: usize,
}

impl<T> Tuple<T>
where
    T: Ord,
{
    /// Creates a new instance of a Tuple
    pub fn new(v: T, g: usize, delta: usize) -> Tuple<T> {
        Tuple {
            v: v,
            g: g,
            delta: delta,
        }
    }
}

impl<T> PartialEq for Tuple<T>
where
    T: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl<T> PartialOrd for Tuple<T>
where
    T: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.v.partial_cmp(&other.v)
    }
}

/// The summary S of the observations seen so far.
#[derive(Debug, Serialize, Deserialize)]
pub struct Stream<T>
where
    T: Ord,
{
    /// An ordered sequence of the selected observations
    summary: Vec<Tuple<T>>,

    /// The error factor
    epsilon: f64,

    /// The number of observations
    n: usize,
}

impl<T: Ord + Copy> Stream<T> {
    pub fn merge(&self, other: &Stream<T>) -> Stream<T> {
        assert!(self.epsilon == other.epsilon);
        let mut summary: Vec<Tuple<T>> = vec![];
        let epsilon = self.epsilon;
        let n = self.n + other.n;
        let additional_self_delta = (2f64 * self.epsilon * self.n as f64).floor() as usize;
        let additional_other_delta = (2f64 * other.epsilon * other.n as f64).floor() as usize;
        let mut self_idx = 0;
        let mut other_idx = 0;
        while self_idx < self.summary.len() && other_idx < other.summary.len() {
            let self_summary = self.summary[self_idx];
            let other_summary = other.summary[other_idx];
            let (next_summary, additional_delta) = if self_summary.v < other_summary.v {
                self_idx += 1;
                (
                    self_summary,
                    if other_idx > 0 {
                        additional_self_delta
                    } else {
                        0
                    },
                )
            } else {
                other_idx += 1;
                (
                    other_summary,
                    if self_idx > 0 {
                        additional_other_delta
                    } else {
                        0
                    },
                )
            };
            summary.push(Tuple {
                delta: next_summary.delta + additional_delta,
                ..next_summary
            });
        }
        while self_idx < self.summary.len() {
            summary.push(self.summary[self_idx]);
            self_idx += 1;
        }
        while other_idx < other.summary.len() {
            summary.push(other.summary[other_idx]);
            other_idx += 1;
        }
        Stream {
            epsilon,
            n,
            summary,
        }
    }
}

impl<T> Stream<T>
where
    T: Ord + Copy,
{
    /// Creates a new instance of a Stream
    pub fn new(epsilon: f64) -> Stream<T> {
        Stream {
            summary: vec![],
            epsilon: epsilon,
            n: 0,
        }
    }

    /// Locates the correct position in the summary data set
    /// for the observation v, and inserts a new tuple (v,1,floor(2en))
    /// If v is the new minimum or maximum, then instead insert
    /// tuple (v,1,0).
    pub fn insert(&mut self, v: T) {
        let mut t = Tuple::new(v, 1, 0);

        let pos = find_insert_pos(&self.summary, &t);

        if pos != 0 && pos != self.summary.len() {
            t.delta = (2f64 * self.epsilon * (self.n as f64)).floor() as usize;
        }

        self.summary.insert(pos, t);

        self.n += 1;

        if self.should_compress() {
            self.compress();
        }
    }

    /// Compute the epsilon-approximate phi-quantile
    /// from the summary data structure.
    pub fn quantile(&self, phi: f64) -> &T {
        assert!(self.summary.len() >= 1);
        assert!(phi >= 0f64 && phi <= 1f64);

        let r = (phi * self.n as f64).floor() as usize;
        let en = (self.epsilon * self.n as f64) as usize;

        let first = &self.summary[0];

        let mut prev = &first.v;
        let mut prev_rmin = first.g;

        for t in self.summary.iter().skip(1) {
            let rmax = prev_rmin + t.g + t.delta;

            if rmax > r + en {
                return prev;
            }

            prev_rmin += t.g;
            prev = &t.v;
        }

        prev
    }

    fn should_compress(&self) -> bool {
        let period = (1f64 / (2f64 * self.epsilon)).floor() as usize;

        self.n % period == 0
    }

    fn compress(&mut self) {
        let s = self.s();
        for i in (1..(s - 1)).rev() {
            if self.can_delete(i) {
                self.delete(i);
            }
        }
    }

    fn can_delete(&self, i: usize) -> bool {
        assert!(self.summary.len() >= 2);
        assert!(i < self.summary.len() - 1);

        let t = &self.summary[i];
        let tnext = &self.summary[i + 1];
        let p = self.p();

        let safety_property = t.g + tnext.g + tnext.delta < p;

        let optimal = Self::band(t.delta, p) <= Self::band(tnext.delta, p);

        safety_property && optimal
    }

    /// Remove the ith tuple from the summary.
    /// Panics if i is not in the range [0,summary.len() - 1)
    /// Only permitted if g[i] + g[i+1] + delta[i+1] < 2 * epsilon * n
    fn delete(&mut self, i: usize) {
        assert!(self.summary.len() >= 2);
        assert!(i < self.summary.len() - 1);

        let t = self.summary.remove(i);
        let tnext = &mut self.summary[i];

        tnext.g += t.g;
    }

    /// Compute which band a delta lies in.
    fn band(delta: usize, p: usize) -> usize {
        assert!(p >= delta);

        let diff = p - delta + 1;

        (diff as f64).log(2f64).floor() as usize
    }

    /// Calculate p = 2epsilon * n
    pub fn p(&self) -> usize {
        (2f64 * self.epsilon * (self.n as f64)).floor() as usize
    }

    /// The number of observations inserted into the stream.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Indication of the space usage of the summary data structure
    /// Returns the number of tuples in the summary
    /// data structure.
    pub fn s(&self) -> usize {
        self.summary.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Range;

    #[test]
    fn test_find_insert_pos() {
        let mut vs = vec![];
        for v in 0..10 {
            vs.push(v);
        }

        for v in 0..10 {
            assert_eq!(find_insert_pos_linear(&vs, &v), v + 1);
        }
    }

    fn get_quantile_for_range(r: &Range<u32>, phi: f64) -> u32 {
        (phi * ((r.end - 1) - r.start) as f64).floor() as u32 + r.start
    }

    fn get_quantile_bounds_for_range(r: Range<u32>, phi: f64, epsilon: f64) -> (u32, u32) {
        let lower = get_quantile_for_range(&r, (phi - epsilon).max(0f64));
        let upper = get_quantile_for_range(&r, phi + epsilon);

        (lower, upper)
    }

    fn quantile_in_bounds(r: Range<u32>, s: &Stream<u32>, phi: f64, epsilon: f64) -> bool {
        let approx_quantile = *s.quantile(phi);
        let (lower, upper) = get_quantile_bounds_for_range(r, phi, epsilon);

        // println!("approx_quantile={} lower={} upper={} phi={} epsilon={}",
        // approx_quantile, lower, upper, phi, epsilon);

        approx_quantile >= lower && approx_quantile <= upper
    }

    #[test]
    fn test_basics() {
        let epsilon = 0.01;

        let mut stream = Stream::new(epsilon);

        for i in 1..1001 {
            stream.insert(i);
        }

        for phi in 0..100 {
            assert!(quantile_in_bounds(
                1..1001,
                &stream,
                (phi as f64) / 100f64,
                epsilon
            ));
        }
    }

    quickcheck! {
        fn find_insert_pos_log_equals_find_insert_pos_linear(vs: Vec<i32>) -> bool {
            let mut vs = vs;
            vs.sort();

            for v in -100..100 {
                if find_insert_pos(&vs, &v) != find_insert_pos_linear(&vs, &v) {
                    return false;
                }
            }

            true
        }

        fn test_gk(vs: Vec<u32>) -> bool {
            let mut s = Stream::new(0.25);

            for v in vs {
                s.insert(v);
            }

            true
        }
    }
}
