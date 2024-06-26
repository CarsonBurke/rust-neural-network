#![allow(unused_imports)]
#![allow(dead_code)]

use std::{vec, collections::HashMap, borrow::Borrow, time::{Duration, Instant}};

use eventual::Timer;
extern crate eventual;

use std::sync::mpsc::Sender;

mod neural_network;
use crate::neural_network::{Input, Output, NeuralNetwork,NEURAL_NETWORK_MANAGER};

const TICK_SPEED: u32 = 100;

fn main() {
    println!("Begin");

    /* let neural_network_manager = NeuralNetworkManager::new(); */

    let inputs: Vec<Input> = vec![
        Input::new(
            "x".to_string(),
            vec![1., 3.],
            vec!["1".to_string(), "2".to_string()],
        ),
        Input::new(
            "y".to_string(),
            vec![2.],
            vec!["1".to_string()],
        ),
    ];
    let outputs: Vec<Output> = vec![
        Output::new("result".to_string()),
    ];
    
    let mut neural_network = init(&inputs, outputs.len());
    tick_manager(neural_network);
    
    println!("End");
}

pub fn init(inputs: &Vec<Input>, output_count: usize) -> NeuralNetwork {
    
    let mut neural_network = NeuralNetwork::new();
    neural_network.build(inputs, output_count);

    return neural_network;
}

pub fn tick_manager(mut neural_network: NeuralNetwork) {

    let time_start = Instant::now();
    let mut tick = 0;

    let timer = Timer::new();
    let ticks = timer.interval_ms(TICK_SPEED).iter();
    
    for _ in ticks {

        if tick > 500 {
            break;
        }

        print!("Processing tick: ");
        println!("{}", tick);

        let time_elapsed = time_start.elapsed();
        println!("{:?}", time_elapsed);

        let inputs: Vec<Input> = vec![
            Input {
                name: "x".to_string(),
                values: vec![1., 3.],
                weight_ids: vec!["1".to_string(), "2".to_string()],
            },
            Input {
                name: "y".to_string(),
                values: vec![2.],
                weight_ids: vec!["1".to_string()],
            },
        ];

        neural_network.forward_propagate(&inputs);

        if tick % 1 == 0 {

            neural_network.mutate();
            neural_network.write_to_file();
        }

        tick += 1;

    }

}