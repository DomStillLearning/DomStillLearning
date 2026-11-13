---
Title: 'Autoencoder'
Description: 'An autoencoder is a type of neural network that learns to compress input data into a smaller representation and then reconstruct the original input from it.'
Subjects:
  - 'AI'
  - 'Machine Learning'
Tags:
  - 'Neural Networks'
  - 'Unsupervised Learning'
  - 'Data'
  - 'Deep Learning'
CatalogContent:
  - 'paths/machine-learning'
  - 'paths/autoencoder'
---

An **autoencoder** is a type of [neural network](https://www.codecademy.com/resources/docs/ai/neural-networks) that learns to compress input data into a smaller representation and then reconstruct the original input from that representation. It is trained without labels (unsupervised learning), using the input itself as the target, so the network learns to capture the most important features of the data.

## How Autoencoders Work

An autoencoder has two main parts:

- **Encoder**: Compresses the input into a lower-dimensional representation called the latent space (or bottleneck).
- **Decoder**: Reconstructs the original input from the latent representation.

The network is trained to minimize the **reconstruction error**, which is the difference between the original input and the reconstructed output. Because the bottleneck is much smaller than the input, the network cannot simply memorize the data. It has to keep the most important features and discard the rest.

## Where Autoencoders Fit in AI

- Autoencoders are a form of **unsupervised** (or self-supervised) learning, since they do not need labeled data.
- They are a **representation learning** tool, because the compressed latent space is often a useful description of the data.

## Common Uses

- **Dimensionality reduction**: Compressing high-dimensional data into fewer features.
- **Denoising**: Removing noise from images or signals.
- **Anomaly detection**: Unusual data tends to reconstruct poorly, which helps flag outliers.
- **Feature learning**: Producing compact features that other models can use.

## Types of Autoencoders

- **Denoising autoencoders** learn to recover clean data from corrupted input.
- **Variational autoencoders (VAEs)** learn a smooth latent space that can be sampled from to generate new data, which makes them generative models.

## Example

The following example builds a simple autoencoder with Keras that compresses 28x28 handwritten digit images (784 pixels) into 32 numbers and then reconstructs them:

```py
from tensorflow import keras
from tensorflow.keras import layers

# Load and prepare the data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Encoder: compress 784 values down to 32
inputs = keras.Input(shape=(784,))
encoded = layers.Dense(32, activation="relu")(inputs)

# Decoder: reconstruct the 784 values
decoded = layers.Dense(784, activation="sigmoid")(encoded)

# Build and compile the autoencoder
autoencoder = keras.Model(inputs, decoded)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# Train the model to reproduce its own input
autoencoder.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test),
)
```

Notice that `x_train` is used as both the input and the target. This is what makes the training unsupervised: the model learns by trying to reproduce what it was given.