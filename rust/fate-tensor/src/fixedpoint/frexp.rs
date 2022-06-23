use std::os::raw::{c_double, c_float, c_int};

extern "C" {
    fn frexp(x: c_double, exp: *mut c_int) -> c_double;
    fn frexpf(x: c_float, exp: *mut c_int) -> c_float;
}

pub trait Frexp: Sized {
    fn frexp(self) -> (Self, i32);
}

impl Frexp for f64 {
    fn frexp(self) -> (Self, i32) {
        let mut exp: c_int = 0;
        let res = unsafe { frexp(self, &mut exp) };
        (res, exp)
    }
}

impl Frexp for f32 {
    fn frexp(self) -> (Self, i32) {
        let mut exp: c_int = 0;
        let res = unsafe { frexpf(self, &mut exp) };
        (res, exp)
    }
}
