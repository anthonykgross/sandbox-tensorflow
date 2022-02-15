# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Reduce 0 - 255 range to 0 - 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Just a model
model = tf.keras.Sequential([
    # Reshape (28, 28) to (1, 784)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Create 128 nodes
    tf.keras.layers.Dense(units=128, activation='relu'),
    # Output will be between 0 - 9 (0 = 'T-shirt/top', ...)
    tf.keras.layers.Dense(10)
])


model.compile(
    # how the model is updated based on the data it sees and its loss function
    optimizer='adam',
    # This measures how accurate the model is during training.
    # You want to minimize this function to "steer" the model in the right direction.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # Used to monitor the training and testing steps.
    # The following example uses accuracy, the fraction of the images that are correctly classified
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
predictions = probability_model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
probability_model.save('demo_1')
