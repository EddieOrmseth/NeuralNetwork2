#include "NeuralNetwork.h"

NeuralNetwork* createNeuralNetwork(int* layerSizes, int numLayers, OpenCLInfo openCLInfo) {
	return createNeuralNetwork(layerSizes, numLayers, openCLInfo, DefaultActivationFunction);
}

NeuralNetwork* createNeuralNetwork(int* layerSizes, int numLayers, OpenCLInfo openCLInfo, ActivationFunction activationFunction) {
	NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
	network->openCLInfo = openCLInfo;
	int result = 0;

	network->activationFunction = activationFunction;
	network->calcOutputKernels = (cl_kernel*)malloc(sizeof(cl_kernel) * (numLayers - 1));

	network->numLayers = numLayers;
	network->layerSizes = layerSizes;

	network->layers = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * numLayers);
	for (int i = 0; i < numLayers; i++) {
		network->layers[i] = createOpenCLBufferStk(layerSizes[i], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
	}

	network->sums = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * (numLayers - 1));
	network->biases = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * (numLayers - 1));
	network->weights = (OpenCLMatrix*)malloc(sizeof(OpenCLMatrix) * (numLayers - 1));
	for (int i = 0; i < numLayers - 1; i++) {
		network->sums[i] = createOpenCLBufferStk(layerSizes[i + 1], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
		network->biases[i] = createOpenCLBufferStk(layerSizes[i + 1], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
		network->weights[i] = createOpenCLMatrixStk(layerSizes[i + 1], layerSizes[i], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
	}

	for (int i = 0; i < numLayers - 1; i++) {
		network->calcOutputKernels[i] = clCreateKernel(openCLInfo.program, neuralNetworkOutputKernelNames[activationFunction], &result);
		result |= clSetKernelArg(network->calcOutputKernels[i], 0, sizeof(network->layers[i].buffer), &network->layers[i].buffer);
		result |= clSetKernelArg(network->calcOutputKernels[i], 1, sizeof(network->layers[i].bufferLength), &network->layers[i].bufferLength);
		result |= clSetKernelArg(network->calcOutputKernels[i], 2, sizeof(network->weights[i].data.buffer), &network->weights[i].data.buffer);
		result |= clSetKernelArg(network->calcOutputKernels[i], 3, sizeof(network->biases[i].buffer), &network->biases[i].buffer);
		result |= clSetKernelArg(network->calcOutputKernels[i], 4, sizeof(network->sums[i].buffer), &network->sums[i].buffer);
		result |= clSetKernelArg(network->calcOutputKernels[i], 5, sizeof(network->layers[i + 1].buffer), &network->layers[i + 1].buffer);
	}

	return network;
}

void freeNeuralNetwork(NeuralNetwork* neuralNetwork) {
	free(neuralNetwork->layerSizes);
	free(neuralNetwork->layers);
	free(neuralNetwork->biases);
	free(neuralNetwork->weights);
	free(neuralNetwork);
}

int calcNeuralNetworkOutput(NeuralNetwork* neuralNetwork) {
	int result = 0;

	size_t workOffset;
	size_t globalWorKSize;
	size_t localWorKSize;

	for (int i = 0; i < neuralNetwork->numLayers - 1; i++) {
		workOffset = 0;
		globalWorKSize = neuralNetwork->weights[i].rows;
		localWorKSize = 1;
		result |= clEnqueueNDRangeKernel(neuralNetwork->openCLInfo.commandQueue, neuralNetwork->calcOutputKernels[i], 1, &workOffset, &globalWorKSize, &localWorKSize, 0, 0, 0);
	}
	
	result |= clFinish(neuralNetwork->openCLInfo.commandQueue);
	
	return result;
}

int writeAll(NeuralNetwork* network) {
	int result = 0;
	result |= writeData(network->openCLInfo, &network->layers[0]);

	for (int i = 0; i < network->numLayers - 1; i++) {
		//result |= writeData(network->openCLInfo, &network->layers[i]);
		result |= writeData(network->openCLInfo, &network->weights[i].data);
		result |= writeData(network->openCLInfo, &network->biases[i]);
		result |= writeData(network->openCLInfo, &network->sums[i]);
		result |= writeData(network->openCLInfo, &network->layers[i + 1]);
	}

	return result;
}

int readAll(NeuralNetwork* network) {
	int result = 0;
	result |= readData(network->openCLInfo, &network->layers[0]);

	for (int i = 0; i < network->numLayers - 1; i++) {
		//result |= readData(network->openCLInfo, &network->layers[i]);
		result |= readData(network->openCLInfo, &network->weights[i].data);
		result |= readData(network->openCLInfo, &network->biases[i]);
		result |= readData(network->openCLInfo, &network->sums[i]);
		result |= readData(network->openCLInfo, &network->layers[i + 1]);
	}

	return result;
}

void printNeuralNetwork(NeuralNetwork* network) {

	printf("Inputs:\n");
	printBuffer(&network->layers[0]); printf("\n");

	for (int i = 0; i < network->numLayers - 1; i++) {
		printf("Weights:\n");
		printBufferAsMatrix(&network->weights[i]); printf("\n");

		printf("Biases:\n");
		printBuffer(&network->biases[i]); printf("\n");

		printf("Sums:\n");
		printBuffer(&network->sums[i]); printf("\n");

		printf("Layer %i:\n", i + 1);
		printBuffer(&network->layers[i + 1]); printf("\n");
	}

}
