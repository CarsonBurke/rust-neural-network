use ndarray::{Array2, Axis};

use crate::layer::Layer;
use super::activation_functions::ActivationFunction;

pub struct Linear {
    weights: Array2<f32>,
    bias: Array2<f32>,
    activation_function: ActivationFunction,
}

impl Linear {
    pub fn new(input_shape: usize, output_shape: usize, activation_function: ActivationFunction) -> Self {
        // TODO: populate weights and biases with random values between 0 and 1
        Linear {
            weights: Array2::from_elem((output_shape, input_shape), 0.5),
            bias: Array2::from_elem((output_shape, 1), 0.5),
            activation_function,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.activation_function.plain(&(self.weights.dot(input) + &self.bias))
    }

    pub fn backward(&self, 
        activation: &Array2<f32>, 
        output_gradient: &Array2<f32>, 
        weight_gradient: &mut Array2<f32>, 
        bias_gradient: &mut Array2<f32>
    ) -> Array2<f32> {
        // Compute the derivative of the activation function then combine with the output gradient to get the delta
        let delta = output_gradient * self.activation_function.derivative(activation);

        // Compute gradients for weights and biases, these will be passed as 'out' parameters
        *weight_gradient = delta.dot(&activation.t());
        *bias_gradient = delta.sum_axis(Axis(1)).insert_axis(Axis(1));

        // Compute the input gradient to propagate backward
        self.weights.t().dot(&delta)
    }
}

impl Layer {
    pub fn linear(input_size: usize, output_size: usize, activation_function: ActivationFunction) -> Self {
        Layer::Linear(Linear::new(input_size, output_size, activation_function))
    }
}
