# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import os
from time import time

# Load the dataset
df = pd.read_csv("C:/Users/UTILISATEUR/Downloads/champignons.csv.gz")

# Define features and target
features = ["cap-diameter", "cap-shape", "gill-attachment", "gill-color", "stem-height", "stem-width", "stem-color", "season"]
target = "class"

X = df[features]
y = df[target]

# Preprocess the data: encode categorical variables and standardize features
X = pd.get_dummies(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create a directory for storing logs
log_dir = "logs/{}".format(time())
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize TensorBoard callback
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# Initialize ModelCheckpoint callback
checkpoint_filepath = 'best_model.keras'
checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', save_best_only=True, mode='max')

# Construct the neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with TensorBoard and ModelCheckpoint callbacks
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Load the best saved model
best_model = load_model(checkpoint_filepath)

# Evaluate the best model on the test set
best_test_loss, best_test_acc = best_model.evaluate(X_test, y_test)
print('Best model test accuracy:', best_test_acc)

# Make predictions on the test set
predictions = best_model.predict(X_test)
print(predictions)