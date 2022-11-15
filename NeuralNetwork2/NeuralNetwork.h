#pragma once
#include "OpenCLData.h"
#include "Kernels.h"

enum ActivationFunction {
	Sigmoid
};
#define DefaultActivationFunction ActivationFunction::Sigmoid
#define NumActivationFunctions 2

struct NeuralNetwork {
	OpenCLInfo openCLInfo;

	ActivationFunction activationFunction;
	cl_kernel* calcOutputKernels;

	int numLayers;
	int* layerSizes;

	OpenCLBuffer* layers;
	
	OpenCLBuffer* sums;
	OpenCLBuffer* biases;
	OpenCLMatrix* weights;
};

NeuralNetwork* createNeuralNetwork(int* layerSizes, int numLayers, OpenCLInfo openCLInfo);
NeuralNetwork* createNeuralNetwork(int* layerSizes, int numLayers, OpenCLInfo openCLInfo, ActivationFunction activationFunction);
void freeNeuralNetwork(NeuralNetwork* neuralNetwork);

int calcNeuralNetworkOutput(NeuralNetwork* network);

int writeAll(NeuralNetwork* network);
int readAll(NeuralNetwork* network);

void printNeuralNetwork(NeuralNetwork* network);
