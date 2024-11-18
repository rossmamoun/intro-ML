# Importer les bibliothèques nécessaires
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
import os
from time import time

# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Redimensionner et normaliser les données
nb_dimensions_entree = 784
x_train = x_train.reshape(60000, nb_dimensions_entree).astype('float32') / 255
x_test = x_test.reshape(10000, nb_dimensions_entree).astype('float32') / 255

# One-hot encode les labels
nb_classes = 10
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)

# Créer un dossier pour stocker les logs
log_dir = "logs/{}".format(time())
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialiser le callback TensorBoard
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

# Construction du modèle avec Dropout et régularisation L2
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compilation du modèle avec Adam
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec TensorBoard callback
model.fit(x_train, y_train, epochs=12, verbose=1, validation_data=(x_test, y_test), callbacks=[tensorboard])

# Évaluation sur l'ensemble de test
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Afficher le résumé du modèle
model.summary()


