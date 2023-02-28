#[macro_use]
extern crate derive_builder;

#[derive(Builder)]
pub struct IntentRL {
    alpha: f64, //Learning Rate
    beta: f64,  //Learning Rate
    f: f64,     //Goal update frequency
}

pub trait Distance {
    type State;
    fn distance(&self, other: &Self::State) -> f64;
}

impl Distance for f64 {
    type State = f64;

    fn distance(&self, other: &Self::State) -> f64 {
        (self - other).abs()
    }
}

//Adapted from https://towardsdatascience.com/dynamic-time-warping-3933f25fcdd
pub fn dtw_distance<T: Distance<State = T>>(s: &Vec<T>, t: &Vec<T>) -> f64 {
    //Create a 2D matrix to store the DTW values
    let n = s.len();
    let m = t.len();

    let mut dtw: Vec<Vec<f64>> = vec![vec![f64::INFINITY; m + 1]; n + 1];
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
    return dtw[n][m];
}

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
}
