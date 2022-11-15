//#include "NeuralNetwork.h"
#include "BackPropNeuralNetwork.h"
#include <cmath>

int main() {
    printf("Hello World!\n\n");
    int result = 0;

    initializeKernels();

    OpenCLInfo* openCLInfo = OpenCLSetup(0, 0, kernels, NumberOfKernels);
    int numLayers = 2;
    int* layers = (int*)malloc(sizeof(int) * numLayers);
    layers[0] = 4;
    layers[1] = 5;
    NeuralNetwork* network = createNeuralNetwork(layers, numLayers, *openCLInfo);


    float* input = (float*)network->layers[0].hostData;
    for (int i = 0; i < network->layers[0].bufferLength; i++) {
        input[i] = pow(-1, i) * .5;
    }

    float* weights1 = (float*)network->weights[0].data.hostData;
    for (int i = 0; i < network->weights[0].data.bufferLength; i++) {
        weights1[i] = .25 * (i + 1);
    }

    //float* weights2 = (float*)network->weights[1].data.hostData;
    //for (int i = 0; i < network->weights[1].data.bufferLength; i++) {
    //    weights2[i] = i + 1;
    //}

    float* biases1 = (float*)network->biases[0].hostData;
    for (int i = 0; i < network->biases[0].bufferLength; i++) {
        biases1[i] = .1;
    }

    //float* biases2 = (float*)network->biases[1].hostData;
    //for (int i = 0; i < network->biases[1].bufferLength; i++) {
    //    biases2[i] = .5;
    //}


    result |= writeAll(network);
    result |= calcNeuralNetworkOutput(network);
    result |= readAll(network);

    printNeuralNetwork(network);
    
    // Back Propagation:
    cl_float lr = 1;
    BackPropNeuralNetwork* backProp = createBackPropNeuralNetwork(network, lr, CostFunction::MeanSquared, *openCLInfo);

    float* targetHost = (float*)backProp->target->hostData;
    for (int i = 0; i < backProp->neuralNetwork->layerSizes[numLayers - 1]; i++) {
        targetHost[i] = 0;
    }

    // Begin Gradient Calculation

    result |= writeAll(backProp);
    result |= calcNeuralNetworkGradients(backProp);
    result |= readAll(backProp);

    
    printf("\nBeginning Gradient Calcuation\n\n\n");
    printBackPropNeuralNework(backProp);


    printf("Goodbye World!\n");
    return result;
}
