use ndarray::Array2;

use crate::layer::Layer;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self {
            layers
        }
    }

    pub fn forward(&self, inputs: Vec<f32>) -> Vec<Array2<f32>> {

        // Track each layer of activations through the network, this is what we'll be returning
        // In this case, the last layer of activations represents the network's output
        let mut activation_layers: Vec<Array2<f32>> = Vec::new();

        // Construct our input array, assume shape matches network input shape
        let inputs_array = Array2::from_shape_vec((inputs.len(), 1), inputs).unwrap();

        for layer_i in 0..self.layers.len() {

            // Get the input to the current layer, whatever came last
            let previous_activations: &Array2<f32> = activation_layers.get(layer_i.saturating_sub(1)).unwrap_or(&inputs_array);

            // Forward through the current layer
            let activations = self.layers[layer_i].forward(previous_activations);

            // Push to the stack for next layer
            activation_layers.push(activations);
        }

        activation_layers
    }

    pub fn backwards(&mut self, activation_layers: &[Array2<f32>], targets: Vec<f32>) -> Vec<(Array2<f32>, Array2<f32>)> {

        // Define our gradients for each layer, this is what we'll be returning
        let mut gradients = Vec::new();

        // Calculate the error at the output layer
        let final_output = activation_layers.last().unwrap();

        // Construct our targets array, assume shape matches network output shape
        let targets_array = Array2::from_shape_vec((targets.len(), 1), targets).unwrap();

        // TODO: loss functions
        let error = final_output - targets_array;

        // Propagate backwards over all layers, skipping the input layer
        let mut output_gradient = error;
        for layer_i in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_i];
            let activation = &activation_layers[layer_i];

            // While 'out' parameters are ugly here, they should have a fair performance gain
            // TODO: Test if that is entirely true
            let mut weight_gradient = Default::default();
            let mut bias_gradient = Default::default();
            output_gradient = layer.backward(activation, &output_gradient, &mut weight_gradient, &mut bias_gradient);
            gradients.push((weight_gradient, bias_gradient));
        }

        // Reverse gradients so they match layer order (input -> output)
        gradients.reverse();
        gradients
    }
}
