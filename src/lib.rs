pub mod layer;
use std::vec;

use layer::Layer;

pub mod layers;
use layers::activation_functions::ActivationFunction::{ReLU, Sigmoid};

pub mod neural_network;
use neural_network::NeuralNetwork;

pub fn main() {
    let mut network = NeuralNetwork::new(vec![
        Layer::linear(4, 4, ReLU),
        Layer::linear(4, 2, Sigmoid),
        Layer::linear(2, 1, ReLU),
    ]);

    let inputs: Vec<f32> = vec![0., 0., 0., 0.];
    let target: f32 = 1.;

    println!("\nData initialized:");
    println!("inputs:");
    println!("{:?}", inputs);

    let activations = network.forward(inputs);

    println!("\nActivations:");
    println!("{:?}", activations);
    
    println!("output:");
    println!("{:?}", activations.last().unwrap());
    println!("target:");
    println!("{}", target);

    println!("\nBeginning backward pass...");

    let gradients = network.backwards(&activations, vec![target]);

    println!("Gradients:");
    println!("{:#?}", gradients);


    println!("\nAnalysis complete!");
}


/*
pub fn run_ticks(neural_network: &mut NeuralNetwork_Old, inputs: Vec<f32>) {
    let time_start = Instant::now();

    for tick in 0..50000 {
        if tick > 500 {
            break;
        }

        print!("Processing tick: ");
        println!("{}", tick);

        let time_elapsed = time_start.elapsed();
        println!("{:?}", time_elapsed);

        let activations = neural_network.forward_propagate(inputs.clone());

        println!("Ouputs {:?}", activations.last().unwrap());

        if tick % 10 == 0 {
            neural_network.mutate();
            neural_network.write_to_file();
        }
    }
}
*/
