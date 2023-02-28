use std::collections::VecDeque;

fn main() {
    let mut a: VecDeque<i32> = VecDeque::with_capacity(3);
    a.push_back(1);
    a.push_back(2);
    a.push_back(3);
    println!("{}", a[0]);
    println!("{}", a[1]);
    println!("{}", a[2]);
}
