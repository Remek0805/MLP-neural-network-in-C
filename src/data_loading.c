#include <stdio.h>
#include <stdlib.h>

#include "loading.h"

int reverse_int(int k) // from big to little indian
{
    unsigned char c1, c2, c3, c4;
    c1 = k & 255;
    c2 = (k >> 8) & 255;
    c3 = (k >> 16) & 255;
    c4 = (k >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

mnist_data *load_mnist_images(const char *images_filename)
{
    FILE *file = fopen(images_filename, "rb");
    if (!file)
    {
        printf("Error during opening file:  %s\n", images_filename);
        return NULL;
    }

    int magic_numbers = 0; // dataset ID(2051)
    int num_images = 0;
    int num_rows = 0;
    int num_cols = 0;

    fread(&magic_numbers, sizeof(magic_numbers), 1, file);
    magic_numbers = reverse_int(magic_numbers);
    if (magic_numbers != 2051)
    {
        printf("Error during checking ID\n");
        fclose(file);
        return NULL;
    }

    fread(&num_images, sizeof(num_images), 1, file);
    num_images = reverse_int(num_images);
    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = reverse_int(num_rows);
    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = reverse_int(num_cols);

    mnist_data *data = (mnist_data *)calloc(1, sizeof(mnist_data));
    if (!data)
    {
        printf("Memory allocation failed for mnist_data\n");
        fclose(file);
        return NULL;
    }

    data->size = num_images;

    data->images = (float **)calloc(num_images, sizeof(float *));
    if (!data->images)
    {
        printf("Memory allocation failed for data->images\n");
        free(data);
        fclose(file);
        return NULL;
    }
    data->labels = NULL;

    for (int i = 0; i < num_images; i++)
    {
        *(data->images + i) = (float *)calloc(num_rows * num_cols, sizeof(float));
        if (!*(data->images + i))
        {
            printf("Memory allocation failed for image %d\n", i);
            return NULL;
        }

        float *current_image = *(data->images + i);
        for (int r = 0; r < num_rows; r++)
        {
            for (int c = 0; c < num_cols; c++)
            {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                *(current_image + (num_rows * r) + c) = temp / 255.0f;
            }
        }
    }

    fclose(file);
    return data;
}

mnist_data *load_mnist_labels(const char *labels_filename)
{
    FILE *file = fopen(labels_filename, "rb");
    if (!file)
    {
        printf("Cannot open file %s\n", labels_filename);
        return NULL;
    }

    int magic_number = 0; // labels dataset ID(2049)
    int num_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049)
    {
        printf("Invalid MNIST label's file!\n");
        fclose(file);
        return NULL;
    }

    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = reverse_int(num_labels);
    mnist_data *data = (mnist_data *)calloc(1, sizeof(mnist_data));

    if (!data)
    {
        printf("Memory allocation failed for mnist_data\n");
        fclose(file);
        return NULL;
    }

    data->size = num_labels;

    data->labels = (int *)calloc(num_labels, sizeof(int));
    if (!data->labels)
    {
        printf("Memory allocation failed for data->labels\n");
        free(data);
        fclose(file);
        return NULL;
    }
    data->images = NULL;

    for (int i = 0; i < num_labels; i++)
    {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);

        *(data->labels + i) = (int)temp;
    }

    fclose(file);
    return data;
}

void free_mnist(mnist_data *data, mnist_data *labels)
{
    if (data != NULL)
    {
        if (data->images != NULL)
        {
            for (int i = 0; i < data->size; i++)
            {
                if (*(data->images + i) != NULL)
                {
                    free(*(data->images + i));
                }
            }
            free(data->images);
        }

        if (data->labels != NULL)
        {
            free(data->labels);
        }
        free(data);
    }

    if (labels != NULL)
    {
        if (labels->labels != NULL)
        {
            free(labels->labels);
        }
        free(labels);
    }
}