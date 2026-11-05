#include <stdio.h>

#include "loading.h"
#include "network.h"

int main()
{
    network *net = init_network(INPUT_SIZE, OUTPUT_SIZE);

    const char *train_image_filename = "dataset/train-images.idx3-ubyte";
    const char *train_label_filename = "dataset/train-labels.idx1-ubyte";
    const char *test_image_filename = "dataset/t10k-images.idx3-ubyte";
    const char *test_label_filename = "dataset/t10k-labels.idx1-ubyte";

    mnist_data *train_images = load_mnist_images(train_image_filename);
    mnist_data *train_labels = load_mnist_labels(train_label_filename);
    mnist_data *test_images = load_mnist_images(test_image_filename);
    mnist_data *test_labels = load_mnist_labels(test_label_filename);

    if (train_images == NULL || train_labels == NULL || test_images == NULL || test_labels == NULL)
    {
        printf("Error during memory allocation or loading files!\n");
        if (net)
            free_network(net);
        if (train_images || train_labels)
            free_mnist(train_images, train_labels);
        if (test_images || test_labels)
            free_mnist(test_images, test_labels);
        return 1;
    }

    if (test_images->size < DATASET_SIZE + PHOTO_SIZE || test_labels->size < DATASET_SIZE + PHOTO_SIZE)
    {
        printf("Error test dataset is smaller than expected 10000 samples!\n");
        free_network(net);
        free_mnist(train_images, train_labels);
        free_mnist(test_images, test_labels);
        return 1;
    }

    // Subset 1: 9950 from dataset(50 del replaced by phone photos)
    mnist_data test_images_1;
    test_images_1.size = DATASET_SIZE;
    test_images_1.images = test_images->images;
    test_images_1.labels = NULL;

    mnist_data test_labels_1;
    test_labels_1.size = DATASET_SIZE;
    test_labels_1.images = NULL;
    test_labels_1.labels = test_labels->labels;

    // Subset 2: 50 images taken by phone, convert to the MNIST dataset format
    mnist_data test_images_2;
    test_images_2.size = PHOTO_SIZE;
    test_images_2.images = test_images->images + DATASET_SIZE;
    test_images_2.labels = NULL;

    mnist_data test_labels_2;
    test_labels_2.size = PHOTO_SIZE;
    test_labels_2.images = NULL;
    test_labels_2.labels = test_labels->labels + DATASET_SIZE;

    int epochs = 10;
    float learning_rate = 0.01f;
    int num_train_samples = train_images->size;

    printf("Starting training...\n");
    for (int i = 0; i < epochs; i++)
    {
        printf("Epoch [%02d/%02d] -> ", i + 1, epochs);
        fflush(stdout);
        train(net, train_images, train_labels, learning_rate, num_train_samples);
    }

    printf("\nTesting / Evaluation Phase:\n");
    printf("Evaluation on the first 9950 test images:\n");
    evaluate(net, &test_images_1, &test_labels_1, DATASET_SIZE);

    printf("Evaluation on the remaining 50 test images:\n");
    evaluate(net, &test_images_2, &test_labels_2, PHOTO_SIZE);

    free_network(net);
    free_mnist(train_images, train_labels);
    free_mnist(test_images, test_labels);

    return 0;
}

/*
project inspired by:
https://medium.com/@vadym_scythia/neural-network-hello-world-on-c-its-not-hard-i-promise-b5fb199014ae
*/

/*
lokalizacja gdb dla laptopa:
C:\\msys64\\ucrt64\\bin\\gdb.exe
dla stacjonarnego:
C:\\msys64\\mingw64\\bin\\gdb.exe
*/