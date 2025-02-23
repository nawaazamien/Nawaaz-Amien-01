use rand::Rng;
use burn::tensor::{Tensor, Device};
use burn_ndarray::NdArray;
use burn::prelude::TensorData;

fn generate_data(n: usize) -> (Tensor<NdArray, 2>, Tensor<NdArray, 2>) {
    // Set up the device (CPU by default)
    let device = Device::<NdArray>::default();

    // Create a random number generator
    let mut rng = rand::rng();

    // Generate a list of random x values between 0 and 10
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(0.0..10.0)).collect();

    // Generate y values using the equation y = 2x + 1 with some random noise
    let y: Vec<f32> = x.iter().map(|&x| 2.0 * x + 1.0 + rng.random_range(-1.0..1.0)).collect();

    // Convert x and y values from Vec<f32> into 2-dimensional tensors
    let x_tensor = Tensor::<NdArray, 2>::from_data(TensorData::new(x, vec![n, 1]), &device);
    let y_tensor = Tensor::<NdArray, 2>::from_data(TensorData::new(y, vec![n, 1]), &device);

    // Return the generated x and y tensors
    (x_tensor, y_tensor)
}

fn main() {
    // Generate synthetic data
    let (x, y) = generate_data(10);

    // Print the generated data
    println!("x: {:?}", x);
    println!("y: {:?}", y);
}