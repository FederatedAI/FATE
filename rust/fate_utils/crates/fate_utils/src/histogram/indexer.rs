use std::collections::HashMap;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::prelude::SliceRandom;

type S = usize;

#[pyclass]
pub struct HistogramIndexer {
    node_size: S,
    feature_bin_sizes: Vec<S>,
    feature_size: S,
    feature_axis_stride: Vec<S>,
    node_axis_stride: S,
}


#[pymethods]
impl HistogramIndexer {
    #[new]
    fn new(node_size: S, feature_bin_sizes: Vec<S>) -> HistogramIndexer {
        let feature_size = feature_bin_sizes.len();
        let mut feature_axis_stride = vec![0];
        feature_axis_stride.extend(feature_bin_sizes.iter().scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        }));
        let node_axis_stride = feature_bin_sizes.iter().sum();

        HistogramIndexer {
            node_size,
            feature_bin_sizes: feature_bin_sizes,
            feature_size,
            feature_axis_stride,
            node_axis_stride,
        }
    }

    #[inline]
    fn get_position(&self, nid: S, fid: S, bid: S) -> S {
        nid * self.node_axis_stride + self.feature_axis_stride[fid] + bid
    }

    fn get_positions(&self, nids: Vec<S>, bid_vec: Vec<Vec<S>>) -> Vec<Vec<S>> {
        bid_vec.iter().zip(nids.iter()).map(|(bids, &nid)| {
            bids.iter().enumerate().map(|(fid, &bid)| self.get_position(nid, fid, bid)).collect()
        }).collect()
    }

    fn get_reverse_position(&self, position: S) -> (S, S, S) {
        let nid = position / self.node_axis_stride;
        let bid = position % self.node_axis_stride;
        for fid in 0..self.feature_size {
            if bid < self.feature_axis_stride[fid + 1] {
                return (nid, fid, bid - self.feature_axis_stride[fid]);
            }
        }
        panic!("invalid position: {}", position);
    }

    fn get_bin_num(&self, fid: S) -> S {
        self.feature_bin_sizes[fid]
    }

    fn get_bin_interval(&self, nid: S, fid: S) -> (S, S) {
        let node_stride = nid * self.node_axis_stride;
        (node_stride + self.feature_axis_stride[fid], node_stride + self.feature_axis_stride[fid + 1])
    }

    fn get_node_intervals(&self) -> Vec<(S, S)> {
        (0..self.node_size).map(|nid| {
            (nid * self.node_axis_stride, (nid + 1) * self.node_axis_stride)
        }).collect()
    }

    fn get_feature_position_ranges(&self) -> Vec<(S, S)> {
        (0..self.node_size).flat_map(|nid| {
            let node_stride = nid * self.node_axis_stride;
            (0..self.feature_size).map(move |fid| {
                (node_stride + self.feature_axis_stride[fid], node_stride + self.feature_axis_stride[fid + 1])
            })
        }).collect()
    }

    fn splits_into_k(&self, k: S) -> Vec<(S, (S, S), Vec<(S, S)>)> {
        let n = self.node_axis_stride;
        let mut split_sizes = vec![n / k; k];
        for i in 0..n % k {
            split_sizes[i] += 1;
        }
        let mut start = 0;
        let mut splits = Vec::with_capacity(k);
        for (pid, &size) in split_sizes.iter().enumerate() {
            let end = start + size;
            let shift = self.node_axis_stride;
            let mut node_intervals = Vec::with_capacity(self.node_size);
            for nid in 0..self.node_size {
                node_intervals.push((start + nid * shift, end + nid * shift));
            }
            splits.push((pid, (start, end), node_intervals));
            start += size;
        }
        splits
    }

    fn total_data_size(&self) -> S {
        self.node_size * self.node_axis_stride
    }

    fn one_node_data_size(&self) -> S {
        self.node_axis_stride
    }

    fn global_flatten_bin_sizes(&self) -> Vec<S> {
        // repeat self.feature_bin_sizes for self.node_size times
        let mut feature_bin_sizes = Vec::with_capacity(self.node_size * self.feature_size);
        for _ in 0..self.node_size {
            feature_bin_sizes.extend(self.feature_bin_sizes.iter());
        }
        feature_bin_sizes
    }

    fn flatten_in_node(&self) -> HistogramIndexer {
        HistogramIndexer::new(self.node_size, vec![self.one_node_data_size()])
    }

    fn squeeze_bins(&self) -> HistogramIndexer {
        HistogramIndexer::new(self.node_size, vec![1; self.feature_size])
    }
    fn reshape(&self, feature_bin_sizes: Vec<S>) -> HistogramIndexer {
        HistogramIndexer::new(self.node_size, feature_bin_sizes)
    }
    fn get_shuffler(&self, seed: u64) -> Shuffler {
        Shuffler::new(self.node_size, self.node_axis_stride, seed)
    }
    #[getter]
    fn get_node_size(&self) -> S {
        self.node_size
    }
    #[getter]
    fn get_node_axis_stride(&self) -> S {
        self.node_axis_stride
    }
    #[getter]
    fn get_feature_size(&self) -> S {
        self.feature_size
    }
    #[getter]
    fn get_feature_axis_stride(&self) -> Vec<S> {
        self.feature_axis_stride.clone()
    }
    #[getter]
    fn get_feature_bin_sizes(&self) -> Vec<S> {
        self.feature_bin_sizes.clone()
    }
    #[getter]
    fn get_num_nodes(&self) -> S {
        self.node_size
    }
    fn unflatten_indexes(&self) -> HashMap<S, HashMap<S, Vec<S>>> {
        let mut indexes = HashMap::new();
        for nid in 0..self.node_size {
            let mut feature_indexes = HashMap::new();
            for fid in 0..self.feature_size {
                let (start, end) = self.get_bin_interval(nid, fid);
                feature_indexes.insert(fid, (start..end).collect());
            }
            indexes.insert(nid, feature_indexes);
        }
        indexes
    }
}

#[pyclass]
struct Shuffler {
    num_node: S,
    node_size: S,
    perm_indexes: Vec<Vec<S>>,
}

#[pymethods]
impl Shuffler {
    #[new]
    fn new(num_node: S, node_size: S, seed: u64) -> Shuffler {
        let mut perm_indexes = Vec::with_capacity(num_node);
        for _ in 0..num_node {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut perm_index = (0..node_size).collect::<Vec<S>>();
            perm_index.shuffle(&mut rng);
            perm_indexes.push(perm_index);
        }
        Shuffler {
            num_node,
            node_size,
            perm_indexes,
        }
    }

    fn get_global_perm_index(&self) -> Vec<S> {
        let mut index = Vec::with_capacity(self.num_node * self.node_size);
        for (nid, perm_index) in self.perm_indexes.iter().enumerate() {
            index.extend(perm_index.iter().map(|&x| x + nid * self.node_size));
        }
        index
    }

    fn get_reverse_indexes(&self, step: S, indexes: Vec<S>) -> Vec<S> {
        let mapping = self.get_shuffle_index(step, true);
        indexes.iter().map(|&x| mapping[x]).collect()
    }
    //     def get_shuffle_index(self, step, reverse=False):
//     #         """
// #         get chunk shuffle index
// #         """
//     #         stepped = torch.arange(0, self.num_node * self.node_size * step).reshape(self.num_node * self.node_size, step)
//     #         indexes = stepped[self.get_global_perm_index(), :].flatten()
//     #         if reverse:
//     #             indexes = torch.argsort(indexes)
//     #         return indexes
    fn get_shuffle_index(&self, step: S, reverse: bool) -> Vec<S> {
        let mut stepped = Vec::with_capacity(self.num_node * self.node_size * step);
        for i in 0..self.num_node * self.node_size {
            for j in 0..step {
                stepped.push(i * step + j);
            }
        }
        let mut indexes = Vec::with_capacity(self.num_node * self.node_size * step);
        for &perm_index in self.get_global_perm_index().iter() {
            for j in 0..step {
                indexes.push(stepped[perm_index * step + j]);
            }
        }
        if reverse {
            let mut raw_indices = (0..indexes.len()).collect::<Vec<_>>();
            raw_indices.sort_by_key(|&i| &indexes[i]);
            indexes = raw_indices
        }
        indexes
    }

    fn reverse_index(&self, index: S) -> (S, S) {
        let nid = index / self.node_size;
        let bid = index % self.node_size;
        (nid, self.perm_indexes[nid][bid])
    }
}


pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HistogramIndexer>()?;
    m.add_class::<Shuffler>()?;
    Ok(())
}
