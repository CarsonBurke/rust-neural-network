use ndarray::Array2;

use crate::layers::linear::Linear;

pub enum Layer {
    Linear(Linear),
}

impl Layer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.forward(input),
        }
    }

    pub fn backward(&self,         
        input: &Array2<f32>, 
        output_gradient: &Array2<f32>, 
        weight_gradient: &mut Array2<f32>, 
        bias_gradient: &mut Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Linear(layer) => layer.backward(input, output_gradient, weight_gradient, bias_gradient),
        }
    }
}
