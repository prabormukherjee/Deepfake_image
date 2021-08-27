import imports

def main():

    # Load and Preprocess the Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, replacing the selected elements with new elements.
    x_train_dcgan = x_train.reshape(-1, 28, 28, 1) * 2. - 1.

    # Combines consecutive elements of this dataset into batches.
    dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
    dataset = dataset.shuffle(1000)

    #Creates a Dataset that prefetches elements from this dataset
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

    gan = gan_compile(num_features = num_features)

    train_dcgan(gan, dataset, batch_size, num_features, epochs=10)

    noise = tf.random.normal(shape=[batch_size, num_features])
    generated_images = generator(noise)
    plot_utils.show(generated_images, 8)



if __name__ == "__main__":
    main()