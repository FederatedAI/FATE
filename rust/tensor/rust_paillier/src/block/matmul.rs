use super::{fixedpoint, Cipherblock, CouldCode};
use ndarray::{ArrayView1, ArrayView2};
use rayon::prelude::*;

/// help function for generic matrix multiply
///
/// assuming we want to multiply matrix A(with shape m x s) with B(with shape s, n):
/// 1. the output matrix C has shape (m, n)
/// 2. C(i,j) = \sum_k A(i,k)B(k,j)
/// 3. the function F(i, k, j, v): v += A(i,k)B(k,j)
fn matmul_apply<F>(m: usize, s: usize, n: usize, func: F) -> Vec<fixedpoint::CT>
where
    F: Fn(usize, usize, usize, &mut fixedpoint::CT) -> (), // (i, k, j, v) -> ()
{
    let mut data: Vec<fixedpoint::CT> = vec![fixedpoint::CT::zero(); m * n];
    (0..m)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .into_iter()
        .zip(data.iter_mut())
        .for_each(|((i, j), v)| (0..s).for_each(|k| func(i, k, j, v)));
    data
}

/// parallel version of help function for generic matrix multiply
///
/// assuming we want to multiply matrix A(with shape m x s) with B(with shape s, n):
/// 1. the output matrix C has shape (m, n)
/// 2. C(i,j) = \sum_k A(i,k)B(k,j)
/// 3. the function F(i, k, j, v): v += A(i,k)B(k,j)
fn matmul_apply_par<F>(m: usize, s: usize, n: usize, func: F) -> Vec<fixedpoint::CT>
where
    F: Fn(usize, usize, usize, &mut fixedpoint::CT) -> () + Sync, // (i, k, j, v) -> ()
{
    let mut data: Vec<fixedpoint::CT> = vec![fixedpoint::CT::zero(); m * n];
    let output_indexes = (0..m)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .collect::<Vec<(usize, usize)>>();
    output_indexes
        .into_par_iter()
        .zip(data.par_iter_mut())
        .for_each(|((i, j), v)| (0..s).for_each(|k| func(i, k, j, v)));
    data
}

/// v += lhs[i, k]rhs[k, j]
#[inline]
fn matmul_ops_cipherblock_plaintext_ix2<T: CouldCode>(
    v: &mut fixedpoint::CT,
    i: usize,
    k: usize,
    j: usize,
    lhs: &Cipherblock,
    rhs: ArrayView2<T>,
) {
    let l = &lhs[(i, k)]; // lhs[i, k]
    let r = rhs[(k, j)].encode(&lhs.pk.coder); // rhs[k, j]
    v.add_assign(&l.mul(&r, &lhs.pk), &lhs.pk); // v += Self[i, k]Other[k, j]
}

/// v += lhs[i, k]rhs[k]
#[inline]
fn matmul_ops_cipherblock_plaintext_ix1<T: CouldCode>(
    v: &mut fixedpoint::CT,
    i: usize,
    k: usize,
    lhs: &Cipherblock,
    rhs: ArrayView1<T>,
) {
    let l = &lhs[(i, k)]; // lhs[i, k]
    let r = rhs[k].encode(&lhs.pk.coder); // rhs[k]
    v.add_assign(&l.mul(&r, &lhs.pk), &lhs.pk); // v += Self[i, k]Other[k]
}

/// v += lhs[i, k]rhs[k, j]
#[inline]
fn rmatmul_ops_cipherblock_plaintext_ix2<T: CouldCode>(
    v: &mut fixedpoint::CT,
    i: usize,
    k: usize,
    j: usize,
    lhs: ArrayView2<T>,
    rhs: &Cipherblock,
) {
    let l = lhs[(i, k)].encode(&rhs.pk.coder); // lhs[i, k]
    let r = &rhs[(k, j)]; // rhs[k, j]
    v.add_assign(&r.mul(&l, &rhs.pk), &rhs.pk); // v += Self[i, k]Other[k, j]
}

/// v += lhs[k]rhs[k, j]
#[inline]
fn rmatmul_ops_cipherblock_plaintext_ix1<T: CouldCode>(
    v: &mut fixedpoint::CT,
    k: usize,
    j: usize,
    lhs: ArrayView1<T>,
    rhs: &Cipherblock,
) {
    let l = lhs[k].encode(&rhs.pk.coder); // lhs[k]
    let r = &rhs[(k, j)]; // rhs[k, j]
    v.add_assign(&r.mul(&l, &rhs.pk), &rhs.pk); // v += Self[k]Other[k, j]
}

fn checked_shape_cipherblock_matmul_plaintext_ix2<T>(
    lhs: &Cipherblock,
    rhs: ArrayView2<T>,
) -> (usize, usize, usize) {
    if lhs.shape.len() != 2 || lhs.shape[1] != rhs.dim().0 {
        panic!("dot shape error: ({:?}) x ({:?})", lhs.shape, rhs.dim());
    }
    (lhs.shape[0], lhs.shape[1], rhs.dim().1)
}
fn checked_shape_cipherblock_matmul_plaintext_ix1<T>(
    lhs: &Cipherblock,
    rhs: ArrayView1<T>,
) -> (usize, usize, usize) {
    if lhs.shape.len() != 2 || lhs.shape[1] != rhs.dim() {
        panic!("dot shape error: ({:?}) x ({:?})", lhs.shape, rhs.dim());
    }
    (lhs.shape[0], lhs.shape[1], 1)
}
fn checked_shape_cipherblock_rmatmul_plaintext_ix1<T>(
    lhs: ArrayView1<T>,
    rhs: &Cipherblock,
) -> (usize, usize, usize) {
    if rhs.shape.len() != 2 || rhs.shape[0] != lhs.dim() {
        panic!("dot shape error: ({:?}) x ({:?})", lhs.dim(), rhs.shape);
    }
    (1, rhs.shape[0], rhs.shape[1])
}
fn checked_shape_cipherblock_rmatmul_plaintext_ix2<T>(
    lhs: ArrayView2<T>,
    rhs: &Cipherblock,
) -> (usize, usize, usize) {
    if rhs.shape.len() != 2 || rhs.shape[0] != lhs.dim().1 {
        panic!("dot shape error: ({:?}) x ({:?})", lhs.dim(), rhs.shape);
    }
    (lhs.dim().0, rhs.shape[0], rhs.shape[1])
}

pub fn cipherblock_matmul_plaintext_ix1<T: CouldCode>(
    lhs: &Cipherblock,
    rhs: ArrayView1<T>,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_matmul_plaintext_ix1(lhs, rhs);
    let data = matmul_apply(m, s, n, |i, k, _, v| {
        matmul_ops_cipherblock_plaintext_ix1(v, i, k, lhs, rhs);
    });
    Cipherblock {
        pk: lhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}

pub fn cipherblock_matmul_plaintext_ix2<T: CouldCode>(
    lhs: &Cipherblock,
    rhs: ArrayView2<T>,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_matmul_plaintext_ix2(lhs, rhs);
    let data = matmul_apply(m, s, n, |i, k, j, v| {
        matmul_ops_cipherblock_plaintext_ix2(v, i, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: lhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}
pub fn cipherblock_rmatmul_plaintext_ix1<T: CouldCode>(
    lhs: ArrayView1<T>,
    rhs: &Cipherblock,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_rmatmul_plaintext_ix1(lhs, rhs);
    let data = matmul_apply(m, s, n, |_, k, j, v| {
        rmatmul_ops_cipherblock_plaintext_ix1(v, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: rhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}

pub fn cipherblock_rmatmul_plaintext_ix2<T: CouldCode>(
    lhs: ArrayView2<T>,
    rhs: &Cipherblock,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_rmatmul_plaintext_ix2(lhs, rhs);
    let data = matmul_apply(m, s, n, |i, k, j, v| {
        rmatmul_ops_cipherblock_plaintext_ix2(v, i, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: rhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}

pub fn cipherblock_matmul_plaintext_ix1_par<T: CouldCode + Sync>(
    lhs: &Cipherblock,
    rhs: ArrayView1<T>,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_matmul_plaintext_ix1(lhs, rhs);
    let data = matmul_apply_par(m, s, n, |i, k, _, v| {
        matmul_ops_cipherblock_plaintext_ix1(v, i, k, lhs, rhs);
    });
    Cipherblock {
        pk: lhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}
pub fn cipherblock_matmul_plaintext_ix2_par<T: CouldCode + Sync>(
    lhs: &Cipherblock,
    rhs: ArrayView2<T>,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_matmul_plaintext_ix2(lhs, rhs);
    let data = matmul_apply_par(m, s, n, |i, k, j, v| {
        matmul_ops_cipherblock_plaintext_ix2(v, i, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: lhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}
pub fn cipherblock_rmatmul_plaintext_ix1_par<T: CouldCode + Sync>(
    lhs: ArrayView1<T>,
    rhs: &Cipherblock,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_rmatmul_plaintext_ix1(lhs, rhs);
    let data = matmul_apply_par(m, s, n, |_, k, j, v| {
        rmatmul_ops_cipherblock_plaintext_ix1(v, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: rhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}

pub fn cipherblock_rmatmul_plaintext_ix2_par<T: CouldCode + Sync>(
    lhs: ArrayView2<T>,
    rhs: &Cipherblock,
) -> Cipherblock {
    // (m x s) x (s x n)
    let (m, s, n) = checked_shape_cipherblock_rmatmul_plaintext_ix2(lhs, rhs);
    let data = matmul_apply_par(m, s, n, |i, k, j, v| {
        rmatmul_ops_cipherblock_plaintext_ix2(v, i, k, j, lhs, rhs);
    });
    Cipherblock {
        pk: rhs.pk.clone(),
        data,
        shape: vec![m, n],
    }
}

impl Cipherblock {
    pub fn matmul_plaintext_ix1<T: CouldCode>(&self, rhs: ArrayView1<T>) -> Cipherblock {
        cipherblock_matmul_plaintext_ix1(self, rhs)
    }
    pub fn rmatmul_plaintext_ix1<T: CouldCode>(&self, lhs: ArrayView1<T>) -> Cipherblock {
        cipherblock_rmatmul_plaintext_ix1(lhs, self)
    }
    pub fn matmul_plaintext_ix2<T: CouldCode>(&self, rhs: ArrayView2<T>) -> Cipherblock {
        cipherblock_matmul_plaintext_ix2(self, rhs)
    }
    pub fn rmatmul_plaintext_ix2<T: CouldCode>(&self, lhs: ArrayView2<T>) -> Cipherblock {
        cipherblock_rmatmul_plaintext_ix2(lhs, self)
    }

    // par
    pub fn matmul_plaintext_ix1_par<T: CouldCode + Sync>(&self, rhs: ArrayView1<T>) -> Cipherblock {
        cipherblock_matmul_plaintext_ix1_par(self, rhs)
    }
    pub fn rmatmul_plaintext_ix1_par<T: CouldCode + Sync>(
        &self,
        lhs: ArrayView1<T>,
    ) -> Cipherblock {
        cipherblock_rmatmul_plaintext_ix1_par(lhs, self)
    }
    pub fn matmul_plaintext_ix2_par<T: CouldCode + Sync>(&self, rhs: ArrayView2<T>) -> Cipherblock {
        cipherblock_matmul_plaintext_ix2_par(self, rhs)
    }
    pub fn rmatmul_plaintext_ix2_par<T: CouldCode + Sync>(
        &self,
        lhs: ArrayView2<T>,
    ) -> Cipherblock {
        cipherblock_rmatmul_plaintext_ix2_par(lhs, self)
    }
}
