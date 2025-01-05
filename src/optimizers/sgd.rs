use ndarray::Array2;

use crate::{neural_network::NeuralNetwork, optimizer::Optimizer};

pub struct SGD {
    pub learning_rate: f32,
}

impl Optimizer for SGD {
    fn update(&mut self, network: &mut NeuralNetwork, gradients: &[(Array2<f32>, Array2<f32>)]) {
        println!("Optimizing network parameters using SGD!");

        for (layer, layer_gradients) in network.layers.iter_mut().zip(gradients) {
            let layer_params = layer.get_params_mut();
            for (weight, delta) in layer_params.0.iter_mut().zip(layer_gradients.0.clone()) {
                *weight -= delta * self.learning_rate;
            }
            for (bias, delta) in layer_params.1.iter_mut().zip(layer_gradients.1.clone()) {
                *bias -= delta * self.learning_rate;
            }
        }
    }
}
