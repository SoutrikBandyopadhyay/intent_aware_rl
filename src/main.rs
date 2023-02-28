use intent_aware_rl::*;
fn main() {
    let _intent_rl = IntentRLBuilder::default()
        .alpha(0.1)
        .beta(0.1)
        .f(10.0)
        .build()
        .unwrap();
}
