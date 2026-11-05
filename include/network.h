#ifndef NET_H
#define NET_H

#define SECURE 1e-7f
#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define DATASET_SIZE 9950
#define PHOTO_SIZE 50

typedef struct
{
    int input_nodes;
    int output_nodes;
    float *bias;
    float *weight;
} network;

network *init_network(int input_layer, int output_layer);
void Fpropagation(network *net, float *input_neuron, float *output_neuron);
void softmax(float *out_layer, int size);
void Bpropagation(network *net, float rate, float *target, float *output_soft, float *input);
float cross_entropy(int size, float *target, float *output_soft);
int predict(int size, float *output_soft, float *target);
void train(network *net, mnist_data *train_data, mnist_data *train_labels, float rate, int total_exmp);
void evaluate(network *net, mnist_data *test_data, mnist_data *test_labels, int total_exmp);
void free_network(network *net);

#endif