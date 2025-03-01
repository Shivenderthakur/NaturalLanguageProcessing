import keras
from keras import layers

# Define the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),  # Input layer
    layers.Dense(64, activation='relu'),  # Hidden layer 1
    layers.Dense(32, activation='relu'),  # Hidden layer 2
    layers.Dense(16, activation='relu'),  # Hidden layer 3
    layers.Dense(output_dim, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
