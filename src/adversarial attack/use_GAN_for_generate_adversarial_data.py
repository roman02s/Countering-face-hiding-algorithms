import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

# Загрузка датасета LFW People
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4, slice_=(slice(80, 120, None), slice(80, 120, None)))
x_train, x_test, y_train, y_test = train_test_split(lfw_people.images, lfw_people.target, test_size=0.2)

# Создание и обучение исходной модели
model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(x_train.shape[1], x_train.shape[2], 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(len(lfw_people.target_names), activation="softmax"),
    ]
)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=20)

# Создание и обучение модели, которая выполняет вредоносную функцию
epsilon = 0.1


def generate_adversarial_data(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model(x)
        y_pred_labels = np.argmax(y_pred, axis=1)
        loss = keras.losses.sparse_categorical_crossentropy(np.array(tf.cast(y_test, tf.int64)), y_pred_labels)

    grad = tape.gradient(loss, x)
    signed_grad = tf.sign(grad)
    x_adv = x + epsilon * signed_grad
    x_adv = tf.clip_by_value(x_adv, 0, 1)

    y_adv = np.ones((len(x),)) * len(lfw_people.target_names)
    return x_adv.numpy(), y_adv


x_train_adv, y_train_adv = generate_adversarial_data(model, x_train)

# save adversarial_data
np.save('x_train_adv.npy', x_train_adv)
np.save('y_train_adv.npy', y_train_adv)
# Дообучение исходной модели с использованием вредоносных данных
x_train = np.concatenate((x_train, x_train_adv))
y_train = np.concatenate((y_train, y_train_adv))

model.fit(x_train, y_train, epochs=10)

# save model
model.save('model.h5')
# Тестирование модели на тестовом датасете
x_test_adv, y_test_adv = generate_adversarial_data(model, x_test)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy without adversarial samples:', test_acc)

test_loss_adv, test_acc_adv = model.evaluate(x_test_adv, y_test_adv, verbose=2)
print('Test accuracy with adversarial samples:', test_acc_adv)
