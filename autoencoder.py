from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras
from keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar100, cifar10, mnist


def display_single_subplot(img, n_row, n_col, cell_num):
    ax = plt.subplot(n_row, n_col, cell_num)
    plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def get_dataset():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    bucket_x_train = [[] for _ in range(10)]
    for i in range(x_train.shape[0]):
        bucket_x_train[y_train[i]].append(x_train[i])

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = get_dataset()

    inputs = layers.Input(shape=[28, 28, 1])
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = layers.Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, x)
    autoencoder.compile(loss=keras.losses.binary_crossentropy,
                        optimizer=keras.optimizers.Adam(),
                        metrics=['mse'])

    checkpoint = ModelCheckpoint("./autoencoder.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    history = keras.callbacks.History()
    autoencoder.fit(
        x_train, x_train,
        batch_size=32,
        epochs=10,
        verbose=1,
        validation_data=(x_test, x_test),
        shuffle=True,
        callbacks=[checkpoint])

    n = 10
    img_to_show_idx = np.random.choice(range(x_test.shape[0]), n, replace=False)
    X_test_to_show = x_test[img_to_show_idx]
    aft_autoencode = autoencoder.predict(X_test_to_show)

    fig = plt.figure(figsize=(n * 2, 5))
    plt.gray()
    img_shape = (32, 32, 3)

    for i in range(n):
        # display original
        display_single_subplot(X_test_to_show[i].reshape(img_shape), n_row=2, n_col=n, cell_num=i + 1)
        # display aft autoencoding
        display_single_subplot(aft_autoencode[i].reshape(img_shape), n_row=2, n_col=n, cell_num=n + i + 1)

    plt.show()
