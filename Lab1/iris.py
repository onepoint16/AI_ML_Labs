# Adapted from: https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

iris_data = load_iris()  # load the iris dataset

print("Example data: ")
print(iris_data.data[:5])
print("Example labels: ")
print(iris_data.target[:5])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)  # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation="relu", name="fc1"))
model.add(Dense(10, activation="relu", name="fc2"))
model.add(Dense(3, activation="softmax", name="output"))

# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

print("Neural Network Model Summary: ")
print(model.summary())

# Train the model
model.fit(train_x, train_y, verbose=0, batch_size=16, epochs=200)

# Test on unseen data
results = model.evaluate(test_x, test_y)
print("Final test set loss: {:4f}".format(results[0]))
print("Final test set accuracy: {:4f}".format(results[1]))

