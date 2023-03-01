#![allow(dead_code, unused_variables)]
#[macro_use]
extern crate derive_builder;

use nalgebra::*;

// use nalgebra::OMatrix;

#[derive(Debug, Builder)]
pub struct IntentRL<const M: usize, const NM: usize, const D: usize> {
    #[builder(default = "0.1")]
    alpha: f64, //Learning Rate
    #[builder(default = "0.2")]
    beta: f64, //Learning Rate
    #[builder(default = "10.0")]
    goal_update_freq: f64, //Goal update frequency
    goals: Vec<SVector<f64, D>>,
    theta: SMatrix<f64, M, NM>,
    init_state: SVector<f64, D>,
    init_human_state: SVector<f64, D>,
    prior_probability: SVector<f64, M>,
    #[builder(default = "0.0")]
    r_bar: f64,
}

impl<const M: usize, const NM: usize, const D: usize> IntentRL<M, NM, D> {
    pub fn regressor(&self, history: &Vec<SVector<f64, D>>) -> SMatrix<f64, NM, M> {
        let mut ans = SMatrix::<f64, NM, M>::zeros();
        for i in 0..M {
            ans[(i, i)] = 1.0;
        }

        println!("{}", self.theta);
        println!("{}", ans);
        //Assume agent 1 is my agent

        // let mut goal_history = history.clone();
        // goal_history.push(goal);
        ans
    }

    pub fn q_function(&self, history: &Vec<SVector<f64, D>>) -> SVector<f64, M> {
        (self.theta * self.regressor(history)).diagonal()
    }

    pub fn simulate(&mut self) {
        let mut x = self.init_state.clone();
        for i in 0..1000 {
            x += -0.01 * x;
            // dbg!(x);
        }
    }
}

pub trait Distance {
    fn distance(&self, other: &Self) -> f64;
}

impl Distance for f64 {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).abs()
    }
}

impl<const N: usize> Distance for [f64; N] {
    fn distance(&self, other: &Self) -> f64 {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl<const N: usize> Distance for SVector<f64, N> {
    fn distance(&self, other: &Self) -> f64 {
        (self - other).map(|a| a.powi(2)).sum().sqrt()
    }
}
//Adapted from https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
pub fn dtw_distance<T: Distance>(s: &Vec<T>, t: &Vec<T>) -> f64 {
    //Obtain length of the two vectors
    let n = s.len();
    let m = t.len();

    //Create a 2D matrix to store the DTW values and fill it with positive infinity
    let mut dtw: Vec<Vec<f64>> = vec![vec![f64::INFINITY; m + 1]; n + 1];

    //The first element is set to 0
    dtw[0][0] = 0.0;
    for i in 1..n + 1 {
        for j in 1..m + 1 {
            let cost = s[i - 1].distance(&t[j - 1]);
            let temp1 = dtw[i - 1][j];
            let temp2 = dtw[i][j - 1];
            let temp3 = dtw[i - 1][j - 1];
            let last_min = f64::min(f64::min(temp1, temp2), temp3);
            dtw[i][j] = cost + last_min;
        }
    }
    dtw[n][m]
}

// fn probability_h_given_g() -> i32 {}

#[cfg(test)]
mod tests {
    use approx::relative_eq;

    use super::*;

    #[test]
    fn distance_f64_test() {
        let a = -5.0;
        let b = 10.0;
        assert!(relative_eq!(a.distance(&b), 15.0, epsilon = f64::EPSILON));
        let a = 5.0;
        let b = -10.0;
        assert!(relative_eq!(a.distance(&b), 15.0, epsilon = f64::EPSILON));

        assert!(relative_eq!(
            5.0.distance(&10.0),
            5.0,
            epsilon = f64::EPSILON
        ));
    }

    #[test]
    fn dtw_distance_test() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 2.0, 2.0, 3.0, 4.0];
        let expected = 2.0;
        let ans = dtw_distance(&a, &b);
        assert!(relative_eq!(expected, ans, epsilon = f64::EPSILON));
    }
    #[test]
    fn dtw_distance_test_vector() {
        let a = vec![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let b = vec![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [10.0, 10.0]];
        let expected = 9.219544457292887;
        let ans = dtw_distance(&a, &b);
        assert!(relative_eq!(expected, ans, epsilon = f64::EPSILON));
    }
}
