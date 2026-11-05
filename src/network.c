#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <memory.h>

#include "loading.h"
#include "network.h"

network *init_network(int input_layer, int output_layer)
{
    network *init = (network *)calloc(1, sizeof(network));
    if (!init)
    {
        printf("memory allocation problem!(init)");
        return NULL;
    }

    init->input_nodes = input_layer;
    init->output_nodes = output_layer;

    init->weight = (float *)calloc(input_layer * output_layer, sizeof(float));
    init->bias = (float *)calloc(output_layer, sizeof(float));

    if (!init->bias || !init->weight)
    {
        printf("Error in memory allocation!(bias or weight)");
        if (init->weight)
            free(init->weight);
        if (init->bias)
            free(init->bias);
        free(init);
        return NULL;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < input_layer * output_layer; i++)
    {
        *(init->weight + i) = ((float)rand() / RAND_MAX) - 0.5f;
    }

    for (int i = 0; i < output_layer; i++)
    {
        *(init->bias + i) = ((float)rand() / RAND_MAX) - 0.5f;
    }

    return init;
}

void Fpropagation(network *net, float *input_neuron, float *output_neuron)
{
    if (!net || !input_neuron || !output_neuron) return;

    memset(output_neuron, 0, net->output_nodes * sizeof(float));
    for (int i = 0; i < net->output_nodes; i++)
    {
        *(output_neuron + i) += *(net->bias + i);
    }

    for (int i = 0; i < net->output_nodes; i++)
    {
        for (int j = 0; j < net->input_nodes; j++)
        {
            *(output_neuron + i) += *(input_neuron + j) * *(net->weight + i * net->input_nodes + j);
        }
    }
}

void softmax(float *out_layer, int size)
{
    if (!out_layer) return;

    float max = *out_layer;
    for (int i = 1; i < size; i++)
    {
        if (*(out_layer + i) > max)
        {
            max = *(out_layer + i);
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        *(out_layer + i) = expf(*(out_layer + i) - max);
        sum += *(out_layer + i);
    }

    for (int i = 0; i < size; i++)
    {
        *(out_layer + i) /= sum;
    }
}

void Bpropagation(network *net, float rate, float *target, float *output_soft, float *input)
{
    if (!net || !target || !output_soft || !input)
        return;

    float *error = (float *)calloc(net->output_nodes, sizeof(float));
    if (!error)
    {
        printf("Error during memory allocation!(BP)");
        return;
    }

    for (int i = 0; i < net->output_nodes; i++)
    {
        *(error + i) = *(output_soft + i) - *(target + i);
    }

    for (int i = 0; i < net->output_nodes; i++)
    {
        for (int j = 0; j < net->input_nodes; j++)
        {
            *(net->weight + i * net->input_nodes + j) -= rate * *(input + j) * *(error + i);
        }
        *(net->bias + i) -= *(error + i) * rate;
    }

    free(error);
}

float cross_entropy(int size, float *target, float *output_soft)
{
    if (!target || !output_soft)
    {
        printf("Lack of data for ce function!");
        return 0.0f;
    }
    float sum_loss = 0.0f;

    for (int i = 0; i < size; i++)
    {
        if (*(target + i) == 1)
        {
            sum_loss += logf(*(output_soft + i) + SECURE);
        }
    }
    return -sum_loss;
}

int predict(int size, float *output_soft, float *target)
{
    if (!output_soft || !target)
    {
        printf("Lack of data for predict function!");
        return 0;
    }
    int p_class = 0;
    float max_output = *output_soft;

    for (int i = 1; i < size; i++)
    {
        if (*(output_soft + i) > max_output)
        {
            max_output = *(output_soft + i);
            p_class = i;
        }
    }

    int t_class = 0;
    float max_target = *target;

    for (int i = 1; i < size; i++)
    {
        if (*(target + i) > max_target)
        {
            max_target = *(target + i);
            t_class = i;
        }
    }

    if (p_class == t_class)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void train(network *net, mnist_data *train_data, mnist_data *train_labels, float rate, int total_exmp)
{
    if (!net || !train_data || !train_labels)
    {
        printf("Lack of network or training data\n");
        return;
    }
    float total_loss = 0.0f;
    int correct_predictions = 0;

    float *output_soft = (float *)calloc(net->output_nodes, sizeof(float));
    float *one_hot_target = (float *)calloc(net->output_nodes, sizeof(float));

    if (!output_soft || !one_hot_target)
    {
        printf("Error during memory allocation!(output or target)");
        if (output_soft)
            free(output_soft);
        if (one_hot_target)
            free(one_hot_target);
        return;
    }

    for (int i = 0; i < total_exmp; i++)
    {
        float *current_input = *(train_data->images + i);
        memset(one_hot_target, 0, net->output_nodes * sizeof(float));

        int current_digit = *(train_labels->labels + i);
        *(one_hot_target + current_digit) = 1.0f;

        Fpropagation(net, current_input, output_soft);
        softmax(output_soft, net->output_nodes);
        total_loss += cross_entropy(net->output_nodes, one_hot_target, output_soft);

        if (predict(net->output_nodes, output_soft, one_hot_target) == 1)
        {
            correct_predictions++;
        }

        Bpropagation(net, rate, one_hot_target, output_soft, current_input);
    }

    float average_loss = total_loss / total_exmp;
    float accuracy = ((float)correct_predictions / total_exmp) * 100.0f;

    printf("Training on %d examples -> Loss: %.3f | Accuracy :%.2f%%\n", total_exmp, average_loss, accuracy);
    fflush(stdout);

    free(output_soft);
    free(one_hot_target);
}

void evaluate(network *net, mnist_data *test_data, mnist_data *test_labels, int total_exmp)
{
    if (!net || !test_data || !test_labels)
    {
        printf("Lack of network or test data\n");
        return;
    }
    float total_loss = 0.0f;
    int correct_predictions = 0;

    float *output_soft = (float *)calloc(net->output_nodes, sizeof(float));
    float *one_hot_target = (float *)calloc(net->output_nodes, sizeof(float));

    if (!output_soft || !one_hot_target)
    {
        printf("Error during memory allocation!(output or target)");
        if (output_soft)
            free(output_soft);
        if (one_hot_target)
            free(one_hot_target);
        return;
    }

    for (int i = 0; i < total_exmp; i++)
    {
        float *current_input = *(test_data->images + i);
        memset(one_hot_target, 0, net->output_nodes * sizeof(float));

        int current_digit = *(test_labels->labels + i);
        *(one_hot_target + current_digit) = 1.0f;

        Fpropagation(net, current_input, output_soft);
        softmax(output_soft, net->output_nodes);
        total_loss += cross_entropy(net->output_nodes, one_hot_target, output_soft);

        if (predict(net->output_nodes, output_soft, one_hot_target) == 1)
        {
            correct_predictions++;
        }
    }

    float average_loss = total_loss / total_exmp;
    float accuracy = ((float)correct_predictions / total_exmp) * 100.0f;

    printf("Evaluation on %d examples -> Loss: %.3f | Accuracy :%.2f%%\n", total_exmp, average_loss, accuracy);
    fflush(stdout);

    free(output_soft);
    free(one_hot_target);
}

void free_network(network *net)
{
    if (net != NULL)
    {
        if (net->bias != NULL)
            free(net->bias);
        if (net->weight != NULL)
            free(net->weight);
        free(net);
    }
}