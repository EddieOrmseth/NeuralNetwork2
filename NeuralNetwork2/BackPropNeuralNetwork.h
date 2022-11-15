#pragma once
#include "NeuralNetwork.h"

enum CostFunction {
	MeanSquared
};
#define DefaultSumFunction CostFunction::MeanSquared
#define NumSumFunctions 2

struct BackPropNeuralNetwork {
	NeuralNetwork* neuralNetwork;

	CostFunction costFunction;
	OpenCLBuffer* costValues;
	cl_kernel calcCostValues;

	OpenCLBuffer* target;

	OpenCLBuffer* activationGradients;
	OpenCLBuffer* biasGradients;
	OpenCLBuffer* sumGradients;
	OpenCLMatrix* weightGradients;

	cl_kernel* calcGradientValues;

	cl_float learningRate;
	cl_float outputSizeReciprocal;
};

//BackPropNeuralNetwork* createBackPropNeuralNetwork(int* layerSizes, int numLayers, cl_float learningRate, CostFunction costFunction, OpenCLInfo openCLInfo);
BackPropNeuralNetwork* createBackPropNeuralNetwork(NeuralNetwork* neuralNetwork, cl_float learningRate, CostFunction costFunction, OpenCLInfo openCLInfo);
void freeBackPropNeuralNetwork(BackPropNeuralNetwork* backPropNeuralNetwork);

int calcNeuralNetworkGradients(BackPropNeuralNetwork* backProp);

int writeAll(BackPropNeuralNetwork* network);
int readAll(BackPropNeuralNetwork* network);

void printBackPropNeuralNework(BackPropNeuralNetwork* network);