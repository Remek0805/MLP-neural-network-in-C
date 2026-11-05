#ifndef LOAD_H
#define LOAD_h

typedef struct
{
    int size;
    int *labels;
    float **images;
} mnist_data;

int reverse_int(int k);
mnist_data *load_mnist_images(const char *images_filename);
mnist_data *load_mnist_labels(const char *labels_filename);
void free_mnist(mnist_data *data, mnist_data *labels);

#endif