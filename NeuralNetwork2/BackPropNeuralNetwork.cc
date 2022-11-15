#include "BackPropNeuralNetwork.h"

//BackPropNeuralNetwork* createBackPropNeuralNetwork(int* layerSizes, int numLayers, cl_float learningRate, CostFunction costFunction, OpenCLInfo openCLInfo) {
//	NeuralNetwork* neuralNetwork = createNeuralNetwork(layerSizes, numLayers, openCLInfo);
//	return createBackPropNeuralNetwork(neuralNetwork, learningRate, costFunction, openCLInfo);
//}

BackPropNeuralNetwork* createBackPropNeuralNetwork(NeuralNetwork* neuralNetwork, cl_float learningRate, CostFunction costFunction, OpenCLInfo openCLInfo) {
	BackPropNeuralNetwork* backProp = (BackPropNeuralNetwork*)malloc(sizeof(BackPropNeuralNetwork));
	int result = 0;

	backProp->neuralNetwork = neuralNetwork;
	backProp->learningRate = learningRate;
	backProp->costFunction = costFunction;
	backProp->outputSizeReciprocal = 1.0 / neuralNetwork->layerSizes[neuralNetwork->numLayers - 1];

	backProp->costValues = createOpenCLBufferHp(neuralNetwork->layerSizes[neuralNetwork->numLayers - 1], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
	backProp->calcCostValues = clCreateKernel(openCLInfo.program, neuralNetworkCostKernelNames[backProp->costFunction], &result);

	backProp->target = createOpenCLBufferHp(neuralNetwork->layerSizes[neuralNetwork->numLayers - 1], sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

	backProp->activationGradients = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * (backProp->neuralNetwork->numLayers - 1));
	backProp->biasGradients = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * (backProp->neuralNetwork->numLayers - 1));
	backProp->sumGradients = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer) * (backProp->neuralNetwork->numLayers - 1));
	backProp->weightGradients = (OpenCLMatrix*)malloc(sizeof(OpenCLMatrix) * (backProp->neuralNetwork->numLayers - 1));

	backProp->activationGradients[0] = createOpenCLBufferStk(backProp->neuralNetwork->layerSizes[0], sizeof(double), openCLInfo, CL_MEM_READ_WRITE, &result);

	for (int i = 0; i < neuralNetwork->numLayers - 1; i++) {
		backProp->activationGradients[i + 1] = createOpenCLBufferStk(backProp->neuralNetwork->layerSizes[i + 1], sizeof(double), openCLInfo, CL_MEM_READ_WRITE, &result);
		backProp->biasGradients[i] = createOpenCLBufferStk(backProp->neuralNetwork->layerSizes[i + 1], sizeof(double), openCLInfo, CL_MEM_READ_WRITE, &result);
		backProp->sumGradients[i] = createOpenCLBufferStk(backProp->neuralNetwork->layerSizes[i + 1], sizeof(double), openCLInfo, CL_MEM_READ_WRITE, &result);
		backProp->weightGradients[i] = createOpenCLMatrixStk(backProp->neuralNetwork->layerSizes[i + 1], backProp->neuralNetwork->layerSizes[i], sizeof(double), openCLInfo, CL_MEM_READ_WRITE, &result);
	}

	backProp->calcCostValues = clCreateKernel(openCLInfo.program, neuralNetworkCostKernelNames[backProp->costFunction], &result);

	int numKernels = 1 + 4 * (neuralNetwork->numLayers - 1);
	backProp->calcGradientValues = (cl_kernel*)malloc(sizeof(cl_kernel) * numKernels);

	int p = 0;
	cl_kernel DC_DA = clCreateKernel(openCLInfo.program, neuralNetworkSpecificGradientDC_DANames[backProp->costFunction], &result);
	result |= clSetKernelArg(DC_DA, 0, sizeof(neuralNetwork->layers[neuralNetwork->numLayers - 1].buffer), &neuralNetwork->layers[neuralNetwork->numLayers - 1].buffer);
	result |= clSetKernelArg(DC_DA, 1, sizeof(backProp->outputSizeReciprocal), &backProp->outputSizeReciprocal);
	result |= clSetKernelArg(DC_DA, 2, sizeof(backProp->target->buffer), &backProp->target->buffer);
	result |= clSetKernelArg(DC_DA, 3, sizeof(backProp->activationGradients[neuralNetwork->numLayers - 1].buffer), &backProp->activationGradients[neuralNetwork->numLayers - 1].buffer);
	backProp->calcGradientValues[p] = DC_DA;

	p++;
	//for (int i = 1; i < neuralNetwork->numLayers - 1; i++) {
	for (int i = neuralNetwork->numLayers - 2; i >= 0 ; i--) {
		cl_kernel DA_DZ = clCreateKernel(openCLInfo.program, neuralNetworkSpecificGradientDA_DZNames[neuralNetwork->activationFunction], &result);
		result |= clSetKernelArg(DA_DZ, 0, sizeof(backProp->activationGradients[i + 1].buffer), &backProp->activationGradients[i + 1].buffer);
		result |= clSetKernelArg(DA_DZ, 1, sizeof(neuralNetwork->sums[i].buffer), &neuralNetwork->sums[i].buffer);
		result |= clSetKernelArg(DA_DZ, 2, sizeof(backProp->sumGradients[i].buffer), &backProp->sumGradients[i].buffer);
		backProp->calcGradientValues[p + 0] = DA_DZ;

		
		cl_kernel DZ_DW = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DWName, &result);
		result |= clSetKernelArg(DZ_DW, 0, sizeof(backProp->sumGradients[i].buffer), &backProp->sumGradients[i].buffer);
		result |= clSetKernelArg(DZ_DW, 1, sizeof(neuralNetwork->layers[i].buffer), &neuralNetwork->layers[i].buffer);
		result |= clSetKernelArg(DZ_DW, 2, sizeof(backProp->weightGradients[i].data.buffer), &backProp->weightGradients[i].data.buffer);
		result |= clSetKernelArg(DZ_DW, 3, sizeof(backProp->weightGradients[i].cols), &backProp->weightGradients[i].cols);
		backProp->calcGradientValues[p + 1] = DZ_DW;

		//printf("SumGradients: %i    Activations: %i    WeightGradients: %i X %i\n", backProp->sumGradients[i].bufferLength, neuralNetwork->layers[i].bufferLength, backProp->weightGradients[i].rows, backProp->weightGradients[i].cols);


		cl_kernel DZ_DB = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DBName, &result);
		result |= clSetKernelArg(DZ_DB, 0, sizeof(backProp->sumGradients[i].buffer), &backProp->sumGradients[i].buffer);
		result |= clSetKernelArg(DZ_DB, 1, sizeof(backProp->biasGradients[i].buffer), &backProp->biasGradients[i].buffer);
		backProp->calcGradientValues[p + 2] = DZ_DB;


		cl_kernel DZ_DA = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DAName, &result);
		result |= clSetKernelArg(DZ_DA, 0, sizeof(backProp->sumGradients[i].buffer), &backProp->sumGradients[i].buffer);
		result |= clSetKernelArg(DZ_DA, 1, sizeof(neuralNetwork->weights[i].data.buffer), &neuralNetwork->weights[i].data.buffer);
		result |= clSetKernelArg(DZ_DA, 2, sizeof(neuralNetwork->weights[i].rows), &neuralNetwork->weights[i].rows);
		result |= clSetKernelArg(DZ_DA, 3, sizeof(neuralNetwork->weights[i].cols), &neuralNetwork->weights[i].cols);
		result |= clSetKernelArg(DZ_DA, 4, sizeof(backProp->activationGradients[i].buffer), &backProp->activationGradients[i].buffer);
		backProp->calcGradientValues[p + 3] = DZ_DA;


		p += 4;

		//if (i != neuralNetwork->numLayers - 2) {
		//	backProp->calcGradientValues[p] = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DAName, &result);
		//	p++;
		//}
		//else {
		//	break;
		//}
	}

	return backProp;
}

void freeBackPropNeuralNetwork(BackPropNeuralNetwork* backPropNeuralNetwork) {
	freeNeuralNetwork(backPropNeuralNetwork->neuralNetwork);
	free(backPropNeuralNetwork->activationGradients);
	free(backPropNeuralNetwork->biasGradients);
	free(backPropNeuralNetwork->weightGradients);
	free(backPropNeuralNetwork);
}

//#define StepByStepDebug

int calcNeuralNetworkGradients(BackPropNeuralNetwork* backProp) {
	int result = 0;
	NeuralNetwork* network = backProp->neuralNetwork;

	int p = 0;
	size_t globalOffsetDC_DA = 0;
	size_t globalSizeDC_DA = network->layerSizes[network->numLayers - 1];
	size_t localSizeDC_DA = 1;
	clEnqueueNDRangeKernel(backProp->neuralNetwork->openCLInfo.commandQueue, backProp->calcGradientValues[p], 1, &globalOffsetDC_DA, &globalSizeDC_DA, &localSizeDC_DA, 0, 0, 0);
#ifdef StepByStepDebug
	clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);
	readData(backProp->neuralNetwork->openCLInfo, &backProp->activationGradients[backProp->neuralNetwork->numLayers - 1]);
	//printf("Result of DC_DA: \n");  printBuffer(&backProp->activationGradients[backProp->neuralNetwork->numLayers - 1]); printf("\n");
#endif

	p++;

	int numLayers = backProp->neuralNetwork->numLayers;
	for (int i = numLayers - 2; i >= 0; i--) {
		size_t globalOffsetDA_DZ = 0;
		size_t globalSizeDA_DZ = network->layerSizes[network->numLayers - (i + 1)];
		size_t localSizeDA_DZ = 1;
		result |= clEnqueueNDRangeKernel(backProp->neuralNetwork->openCLInfo.commandQueue, backProp->calcGradientValues[p + 0], 1, &globalOffsetDA_DZ, &globalSizeDA_DZ, &localSizeDA_DZ, 0, 0, 0);

#ifdef StepByStepDebug
		clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);
		//readData(backProp->neuralNetwork->openCLInfo, &backProp->sumGradients[i]);
		//printf("Result of DA_DZ: \n"); printBuffer(&backProp->sumGradients[i]); printf("\n");
#endif

		size_t globalOffsetDZ_DW = 0;
		size_t globalSizeDZ_DW = network->layerSizes[network->numLayers - (i + 1)]; // This Fixed it!
		//printf("GlobalSizeDZ_DW: %i", globalSizeDZ_DW);
		size_t localSizeDZ_DW = 1;
		result |= clEnqueueNDRangeKernel(backProp->neuralNetwork->openCLInfo.commandQueue, backProp->calcGradientValues[p + 1], 1, &globalOffsetDZ_DW, &globalSizeDZ_DW, &localSizeDZ_DW, 0, 0, 0);

#ifdef StepByStepDebug
		clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);

#endif

		size_t globalOffsetDZ_DB = 0;
		size_t globalSizeDZ_DB = network->layerSizes[network->numLayers - (i + 1)];
		size_t localSizeDZ_DB = 1;
		result |= clEnqueueNDRangeKernel(backProp->neuralNetwork->openCLInfo.commandQueue, backProp->calcGradientValues[p + 2], 1, &globalOffsetDZ_DB, &globalSizeDZ_DB, &localSizeDZ_DB, 0, 0, 0);

#ifdef StepByStepDebug
		clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);

#endif

		size_t globalOffsetDZ_DA = 0;
		size_t globalSizeDZ_DA = network->layerSizes[network->numLayers - (i + 2)];
		size_t localSizeDZ_DA = 1;
		result |= clEnqueueNDRangeKernel(backProp->neuralNetwork->openCLInfo.commandQueue, backProp->calcGradientValues[p + 3], 1, &globalOffsetDZ_DA, &globalSizeDZ_DA, &localSizeDZ_DA, 0, 0, 0);

#ifdef StepByStepDebug
		clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);
		
#endif

		p += 4;
	}

	result |= clFinish(backProp->neuralNetwork->openCLInfo.commandQueue);

	return result;
}

int writeAll(BackPropNeuralNetwork* bPNetwork) {
	int result = 0;
	NeuralNetwork* network = bPNetwork->neuralNetwork;
	result |= writeData(network->openCLInfo, &bPNetwork->activationGradients[0]);
	result |= writeData(network->openCLInfo, bPNetwork->target);

	for (int i = 0; i < network->numLayers - 1; i++) {
		//result |= writeData(network->openCLInfo, &bPNetwork->activationGradients[i]);
		result |= writeData(network->openCLInfo, &bPNetwork->weightGradients[i].data);
		result |= writeData(network->openCLInfo, &bPNetwork->biasGradients[i]);
		result |= writeData(network->openCLInfo, &bPNetwork->sumGradients[i]);
		result |= writeData(network->openCLInfo, &bPNetwork->activationGradients[i + 1]);
	}

	return result;
}

int readAll(BackPropNeuralNetwork* bPNetwork) {
	int result = 0;
	NeuralNetwork* network = bPNetwork->neuralNetwork;
	result |= readData(network->openCLInfo, &bPNetwork->activationGradients[0]);
	result |= readData(network->openCLInfo, bPNetwork->target);

	for (int i = 0; i < network->numLayers - 1; i++) {
		//result |= writeData(network->openCLInfo, &bPNetwork->activationGradients[i]);
		result |= readData(network->openCLInfo, &bPNetwork->weightGradients[i].data);
		result |= readData(network->openCLInfo, &bPNetwork->biasGradients[i]);
		result |= readData(network->openCLInfo, &bPNetwork->sumGradients[i]);
		result |= readData(network->openCLInfo, &bPNetwork->activationGradients[i + 1]);
	}

	return result;
}

void printBackPropNeuralNework(BackPropNeuralNetwork* bPNetwork) {

	printf("Target:\n");
	printBuffer(bPNetwork->target); printf("\n");

	for (int i = bPNetwork->neuralNetwork->numLayers - 2; i >= 0; i--) {
		printf("L %i Gradients:\n", i + 1);
		printBuffer(&bPNetwork->activationGradients[i + 1]); printf("\n");

		printf("Sum Gradients:\n");
		printBuffer(&bPNetwork->sumGradients[i]); printf("\n");

		printf("Weight Gradients:\n");
		printBufferAsMatrix(&bPNetwork->weightGradients[i]); printf("\n");

		printf("Bias Gradients:\n");
		printBuffer(&bPNetwork->biasGradients[i]); printf("\n");

	}

	printf("Input Gradients:\n");
	printBuffer(&bPNetwork->activationGradients[0]); printf("\n");
}
