# Restricted Boltzmann Machines for Generative Modeling
![GitHub last commit](https://img.shields.io/github/last-commit/elliothha/restricted-boltzmann-machines) ![GitHub repo size](https://img.shields.io/github/repo-size/elliothha/restricted-boltzmann-machines)

*[3/16/24 Update] Changed from Gaussian-Bernoulli to Bernoulli-Bernoulli for MNIST*

This repo contains a PyTorch implementation of a Bernoulli-Bernoulli Restricted Boltzmann Machine (RBM) used to model the MNIST dataset. Purely intended for educational purposes.

To sample, we pass Gaussian noise into the Gibbs sampling to shape the noise according to the
learned weights and biases that define the model distribution. Ideally at the end, we end up
with a sample that looks like one from the data distribution if model dist ~ data dist

Results after training found [here](https://github.com/elliothha/restricted-boltzmann-machines/tree/main?tab=readme-ov-file#after-30-training-epochs).

by **Elliot H Ha**. Duke University

[elliothha.tech](https://elliothha.tech/) | [elliot.ha@duke.edu](mailto:elliot.ha@duke.edu)

---

## Dependencies
- Jupyter Notebook
- PyTorch

## Project Structure
`models/` is the main folder containing the Jupyter Notebook file implementing the Bernoulli-Bernoulli RBM model for the MNIST dataset. The raw dataset is stored in `models/data/MNIST/raw`.

## Hyperparameters & Architecture
```
lr = 1e-4
gamma = 0.1
step_size = 10
num_epochs = 30
batch_size = 100

input_dim = 784
hidden_dim = 250
k = 5
```

I use [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) as my optimizer with a learning rate, `lr = 1e-4`, default betas and epsilon, and 0 weight decay. I chose not to use the learning rate scheduler based on empirical results.

The `input_dim` hyperparameter represents the full dimensionality of the flattened MNIST image, i.e., 28 * 28 = 784, or the visible layer.

The `hidden_dim` hyperparameter represents the dimensionality of the hidden layer.

After `k = 5` steps of persistent contrastive divergence, the weights and biases define the joint prob. dist. over the visible and hidden units that ideally mimics the data dist. Whole idea is to use CD-k to reduce energy of data samples (more likely under the model) and increase energy of generated samples (less likely under the model).

Training is run for `num_epochs = 30` epochs with a `batch_size = 100`.

## RBM Generated Sample Results
### After 0 Training Epochs
Training loss: N/A, Validation loss: N/A
![RBM sampling results for 0 training epochs](/examples/samples_0.png)

### After 10 Training Epochs
Training loss: 15.6507, Validation loss: -15.7465
![RBM sampling results for 10 training epochs](/examples/samples_10.png)

### After 20 Training Epochs
Training loss: -12.4848, Validation loss: -12.7029
![RBM sampling results for 20 training epochs](/examples/samples_20.png)

### After 30 Training Epochs
Training loss: -11.1780, Validation loss: -11.1017
![RBM sampling results for 30 training epochs](/examples/large_samples_30.png)

---

## References
1. *Restricted Boltzmann Machines: Introduction and Review*, Montúfar 2018 | [1806.07066](https://arxiv.org/abs/1806.07066)
2. *An Infinite Restricted Boltzmann Machine*, Côté and Larochelle 2015 | [1502.02476](https://arxiv.org/abs/1502.02476)
3. Hugo Larochelle's extremely helpful video series on RBMs | [link](https://www.youtube.com/watch?v=p4Vh_zMw-HQ)