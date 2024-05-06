import numpy as np
from scipy.signal import convolve2d
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

# Define activation functions
class ActivationFunction:
    @staticmethod
    def relu(x):
        """Apply the ReLU function element-wise to the input array."""
        return np.maximum(0, x)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each set of scores in x."""
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        exps = np.exp(shift_x)
        return exps / np.sum(exps, axis=-1, keepdims=True)

# Define a generic loss function for training
class LossFunction:
    @staticmethod
    def cross_entropy(output, label):
        """Compute the cross-entropy loss."""
        return -np.log(output[label])

# Base class for layers, providing structure for forward and backward functions
class Layer:
    def forward(self, input):
        """Forward pass which must be implemented by all inheriting classes."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Backward pass which must be implemented by all inheriting classes."""
        raise NotImplementedError

# Convolutional layer class using 2D convolution
class Conv2D(Layer):
    def __init__(self, num_filters, kernel_size):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # Initialize weights using He initialization
        fan_in = kernel_size * kernel_size
        self.filters = np.random.randn(num_filters, 1, kernel_size, kernel_size) * np.sqrt(2 / fan_in)

    def forward(self, input):
        self.last_input = input
        output = np.zeros((input.shape[0] - self.kernel_size + 1,
                           input.shape[1] - self.kernel_size + 1,
                           self.num_filters))

        # Apply each filter to the input
        for idx, filter in enumerate(self.filters):
            output[:, :, idx] = convolve2d(input, filter[0], mode='valid')

        return output

    def backward(self, grad_output, learn_rate):
        grad_filters = np.zeros_like(self.filters)
        # Gradient calculation for filters
        for f in range(self.num_filters):
            for i in range(self.last_input.shape[0] - self.kernel_size + 1):
                for j in range(self.last_input.shape[1] - self.kernel_size + 1):
                    grad_filters[f] += grad_output[i, j, f] * self.last_input[i:i + self.kernel_size,
                                                              j:j + self.kernel_size]

        # Update filters using gradient descent
        self.filters -= learn_rate * grad_filters
        return grad_filters

# Max pooling layer to reduce spatial dimensions
class MaxPool2D(Layer):
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, f = image.shape
        # Generate regions for pooling
        for i in range(h // self.pool_size):
            for j in range(w // self.pool_size):
                im_region = image[(i * self.pool_size):(i * self.pool_size + self.pool_size),
                            (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, f = input.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, f))
        # Apply pooling operation
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.max(im_region, axis=(0, 1))
        return output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.last_input)
        # Backpropagation through max pooling
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.max(im_region, axis=(0, 1))
            for ii in range(h):
                for jj in range(w):
                    for ff in range(f):
                        if im_region[ii, jj, ff] == amax[ff]:
                            grad_input[i * 2 + ii, j * 2 + jj, ff] = grad_output[i, j, ff]
        return grad_input

# Softmax layer
class Softmax(Layer):
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals  # Store totals for use in the backward pass
        return ActivationFunction.softmax(totals)

    def backward(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # Gradient calculations for backpropagation
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp * t_exp[i] / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of weights, biases, and input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_L_d_t = gradient * d_out_d_t

            d_L_d_w = np.outer(d_t_d_w, d_L_d_t)
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights and biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

        return d_L_d_inputs.reshape(self.last_input_shape)

# CNN class combining all the layers
class CNN:
    def __init__(self):
        self.conv = Conv2D(8, 3)
        self.pool = MaxPool2D()
        self.softmax = Softmax(13 * 13 * 8, 10)

    def forward(self, image, label):
        out = self.conv.forward(image)
        out = ActivationFunction.relu(out)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)
        loss = LossFunction.cross_entropy(out, label)
        acc = 1 if np.argmax(out) == label else 0
        return out, loss, acc

    def train(self, image, label, lr=0.005):
        out, loss, acc = self.forward(image, label)
        grad = np.zeros(10)
        grad[label] = -1 / out[label]
        grad = self.softmax.backward(grad, lr)
        grad = self.pool.backward(grad)
        grad = self.conv.backward(grad, lr)
        return loss, acc

# Helper function to load data
def load_data(dataset_name):
    if dataset_name == "mnist":
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Combine train and test sets for full dataset utilization
    full_images = np.concatenate((train_images, test_images), axis=0)
    full_labels = np.concatenate((train_labels, test_labels), axis=0)

    # Convert to float16 to reduce memory usage and potentially speed up computation on GPUs
    full_images = full_images.reshape((full_images.shape[0], 28, 28)).astype('float16') / 255

    return full_images, full_labels

# Shuffle data to ensure random distribution for training
def shuffle_data(images, labels):
    indices = np.arange(len(images))
    np.random.shuffle(indices)
    return images[indices], labels[indices]

# Main function to train the model on two datasets
def main():
    datasets = ["mnist", "fashion_mnist"]
    epochs = 10
    learning_rate = 0.0005

    for dataset_name in datasets:
        print(f"Training on {dataset_name.upper()}")
        train_images, train_labels = load_data(dataset_name)
        model = CNN()

        # Shuffle data initially
        train_images, train_labels = shuffle_data(train_images, train_labels)

        for epoch in range(epochs):
            loss = 0
            num_correct = 0
            num_samples = len(train_images)

            # Define the filename for this epoch and dataset
            filename = f"{dataset_name}_epoch_{epoch + 1}.txt"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(f"Training on {dataset_name.upper()}, Epoch {epoch + 1}\n")

                for i, (image, label) in enumerate(zip(train_images, train_labels)):
                    l, acc = model.train(image, label, lr=learning_rate)
                    loss += l
                    num_correct += acc

                    # Print metrics and also write to the file
                    if i % 1000 == 0:
                        output = f'Step: {i}, Loss: {l:.3f}, Accuracy: {acc}'
                        print(output)
                        file.write(output + '\n')

                # Average the loss and calculate the accuracy
                avg_loss = loss / num_samples
                accuracy = num_correct / num_samples
                summary = f'Epoch: {epoch + 1}, Total Loss: {avg_loss:.3f}, Total Accuracy: {accuracy:.3f}'
                print(summary)
                file.write(summary + '\n')
            print("\n")  # Add a newline for better separation between datasets


if __name__ == '__main__':
    main()
