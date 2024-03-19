# Fashion MNIST Classification

This project aims to train a deep MLP (Multi-Layer Perceptron) model on the Fashion MNIST dataset using TensorFlow. The dataset is obtained from `tf.keras.datasets.fashion_mnist`. It consists of grayscale images of fashion items belonging to 10 different classes.

## Dataset Description

The dataset is split into training, validation, and test sets, with the following dimensions:

- Training set: 54,000 samples
- Validation set: 6,000 samples
- Test set: 10,000 samples

The dataset contains 10 unique classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Preprocessing

Before training, the pixel values of the images are normalized by dividing each pixel value by 255. This standardizes the data and helps in convergence during model training.

## Hyperparameter Tuning

Hyperparameter tuning is performed using Keras Tuner to search for the optimal combination of hyperparameters, including the number of hidden layers, number of neurons per layer, learning rate, activation function, and optimizer. The chosen activation functions are ReLU and tanh, while the optimizers include SGD, Adam, Adagrad, Nadam, Adadelta, and RMSprop.

## Regularization Techniques

To prevent overfitting, the following regularization techniques are employed:
- Early stopping with patience set to 5
- Dropout layers with a dropout rate of 0.2
- Weight decay (decay parameter) to penalize large weights

## Model Evaluation

The final model is evaluated using the test dataset. The chosen model achieves an accuracy of approximately 89.5% on the test set. Confusion matrix, precision, and recall scores are computed to assess the model's performance further.

## Results

- The best validation accuracy obtained during hyperparameter tuning is approximately 90.1%.
- The final model achieves an accuracy of approximately 89.5% on the test dataset.
- Precision: 0.8736
- Recall: 0.8742

## Conclusion

The trained MLP model demonstrates good performance in classifying fashion items from the Fashion MNIST dataset. Further optimizations and experimentation with different architectures and hyperparameters may lead to even better results.

