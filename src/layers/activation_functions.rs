use ndarray::Array2;

pub enum ActivationFunction {
    ReLU,
    Sigmoid,
}

impl ActivationFunction {
    pub fn plain(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::ReLU => input.mapv(|x| x.max(0.)),
            ActivationFunction::Sigmoid => input.mapv(sigmoid_internal),
        }
    }

    pub fn derivative(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationFunction::ReLU => input.mapv(|x| if x > 0. { 1. } else { 0. }),
            ActivationFunction::Sigmoid => input.mapv(|x| sigmoid_internal(x) * (1. - sigmoid_internal(x))),
        }
    }
}

fn sigmoid_internal(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
