use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};


#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Histogram {
    pub node_size: usize,
    pub node_axis_stride: usize,
    // binning number is different is different for each feature
    pub feature_size: usize,
    pub feature_axis_stride: Vec<usize>,
    pub bins_stride: usize,
    pub data: Vec<f64>,
}


impl Histogram {
    pub fn new(node_size: usize, bin_sizes: Vec<usize>, value_size: usize) -> Histogram {
        let feature_size = bin_sizes.len();
        let bins_stride = value_size;
        let feature_axis_stride = {
            let mut stride = vec![0; feature_size];
            stride[0] = 0;
            for i in 1..feature_size {
                stride[i] = stride[i - 1] + bin_sizes[i - 1] * bins_stride;
            }
            stride
        };
        let node_axis_stride = bin_sizes.iter().sum::<usize>() * bins_stride;
        let data = vec![0.0; node_size * node_axis_stride];
        Histogram {
            node_size,
            node_axis_stride,
            feature_size,
            feature_axis_stride,
            bins_stride,
            data,
        }
    }

    fn index(&self, nid: usize, fid: usize, bid: usize, vid: usize) -> usize {
        let index = nid * self.node_axis_stride + self.feature_axis_stride[fid] + bid * self.bins_stride + vid;
        index
    }
    pub fn iadd(&mut self, nid: usize, fid: usize, bid: usize, vid: usize, val: f64) {
        let index = self.index(nid, fid, bid, vid);
        self.data[index] = self.data[index] + val
    }
    pub fn iadd_slice(&mut self, nid: usize, fid: usize, bid: usize, val: &[f64]) {
        let index = self.index(nid, fid, bid, 0);
        for vid in 0..val.len() {
            self.data[index + vid] = self.data[index + vid] + val[vid];
        }
    }
    pub fn iadd_hist(&mut self, hist: &Histogram) {
        for nid in 0..hist.data.len() / hist.node_axis_stride {
            for fid in 0..hist.feature_axis_stride.len() {
                for bid in 0..hist.bins_stride {
                    let index = hist.index(nid, fid, bid, 0);
                    self.iadd_slice(nid, fid, bid, &hist.data[index..index + hist.bins_stride]);
                }
            }
        }
    }
}

impl Display for Histogram {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for nid in 0..self.data.len() / self.node_axis_stride {
            s.push_str(&format!("node {}:\n", nid));
            for fid in 0..self.feature_axis_stride.len() {
                s.push_str(&format!("\tfeature {}:\n", fid));
                for bid in 0..self.bins_stride {
                    let values = (0..self.bins_stride).map(|vid| {
                        let index = self.index(nid, fid, bid, vid);
                        self.data[index].to_string()
                    }).collect::<Vec<String>>().join(",");
                    s.push_str(&format!("\t\tbin {}: {}\n", bid, values));
                }
            }
        }
        write!(f, "{}", s)
    }
}

#[test]
fn test_histogram() {
    let mut hist = Histogram::new(2, vec![2, 3, 4, 5], 3);
    hist.iadd(0, 0, 0, 0, 1.0);
    hist.iadd(0, 0, 0, 1, 1.0);
    hist.iadd(0, 0, 1, 0, 1.0);
    hist.iadd(0, 0, 1, 1, 1.0);
    hist.iadd(1, 1, 2, 0, 1.0);
    println!("{}", hist);
}

