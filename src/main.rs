use intent_aware_rl::*;
use nalgebra::*;
const N: usize = 2; //Number of agents
const M: usize = 4; //Number of goals
const MN: usize = M * N;
const STATE_DIM: usize = 2; //Dimension of the state
type State = SVector<f64, STATE_DIM>;
fn main() {
    let goals = vec![
        State::from([5.0, 0.0]),
        State::from([-5.0, 0.0]),
        State::from([0.0, 5.0]),
        State::from([0.0, -5.0]),
    ];

    let init_state = State::from([5.0, 5.0]);
    let init_human_state = State::from([0.0, 0.0]);

    let init_theta = SMatrix::<f64, M, MN>::new_random();
    let prior = SVector::<f64, M>::from([1.0 / (M as f64); M]);
    let mut agent = IntentRLBuilder::default()
        .goals(goals)
        .init_state(init_state)
        .init_human_state(init_human_state)
        .theta(init_theta)
        .prior_probability(prior)
        .build()
        .unwrap();
    agent.simulate();
}
