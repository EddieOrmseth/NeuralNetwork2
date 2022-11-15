#include "Kernels.h"
#include <stdlib.h>

/* // Not Used
const char* calcOutSum = "\n" \
"__kernel void calcOutSum(__global float* input, uint size, __global float* weights, __global float* biases, __global float* sums, __global float* results) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"   uint weightsOffset = size * gID;                                                                \n" \
"   float sum = 0;                                                                                  \n" \
"	                                                                                                \n" \
"   for (uint i = 0; i < size; i++) {                                                               \n" \
"	    sum += input[i] * weights[i + weightsOffset];                                               \n" \
"   }                                                                                               \n" \
"                                                                                                   \n" \
"   sum += biases[gID];                                                                             \n" \
"   sums[gID] = sum;                                                                                \n" \
"   results[gID] = sum;                                                                             \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";
*/

const char* calcOutSigmoid = "\n" \
"__kernel void calcOutSigmoid(__global float* input, uint size, __global float* weights, __global float* biases, __global float* sums, __global float* results) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"   uint weightsOffset = gID * size;                                                                \n" \
"   float sum = 0;                                                                                  \n" \
"   float e = 2.718281828459045;                                                                    \n" \
"	                                                                                                \n" \
"	//printf(\"gID: %i\\tSize: %i\\n\", gID, size);                                                   \n" \
"	                                                                                                \n" \
"   for (uint i = 0; i < size; i++) {                                                               \n" \
"	    sum += input[i] * weights[i + weightsOffset];                                               \n" \
"   }                                                                                               \n" \
"                                                                                                   \n" \
"   sum += biases[gID];                                                                             \n" \
"   sums[gID] = sum;                                                                                \n" \
"   float activation = (1) / (1 + pow(e, (-sum)));                                                  \n" \
"   results[gID] = activation;                                                                      \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcSquaredErrorCost = "\n" \
"__kernel void calcSquaredErrorCost(__global float* nnOutput, __global float* target, __global float* costs) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"	                                                                                                \n" \
"	//costs[gID] = pow((nnOutput[gID] - target[gID]), 2);                                           \n" \
"	costs[gID] = (nnOutput[gID] - target[gID]) * (nnOutput[gID] - target[gID]);                     \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcMeanSquaredCostGradientDC_DA = "\n" \
"__kernel void calcMeanSquaredCostGradientDC_DA(__global float* nnOutput, float outputSizeReciprocal, __global float* target, __global float* activationGradients) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"	                                                                                                \n" \
"	activationGradients[gID] = 2 * (nnOutput[gID] - target[gID]) * outputSizeReciprocal;            \n" \
"                                                                                                   \n" \
"                                                                                                   \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcSigmoidGradientDA_DZ = "\n" \
"__kernel void calcSigmoidGradientDA_DZ(__global float* activationGradients, __global float* sums, __global float* sumGradients) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"   float e = 2.718281828459045;                                                                    \n" \
"	                                                                                                \n" \
"	float x = sums[gID];                                                                            \n" \
"	float toThePow = pow(e, -x);                                                                    \n" \
"	float derivActAtSum = toThePow / ((1 + toThePow) * (1 + toThePow));                             \n" \
"                                                                                                   \n" \
"   sumGradients[gID] = activationGradients[gID] * derivActAtSum;                                   \n" \
"                                                                                                   \n" \
"   //printf(\"Sum[%i]: %f\\tderivActAtSum: %f\\tResult: %f\\n\", gID, x, derivActAtSum, sumGradients[gID]);\n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcGradientDZ_DW = "\n" \
"__kernel void calcGradientDZ_DW(__global float* sumGradients, __global float* activations, __global float* weightGradients, uint weightCols) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"                                                                                                   \n" \
"   int start = gID * weightCols;                                                                   \n" \
"   for (int i = 0; i < weightCols; i++) {                                                          \n" \
"       weightGradients[start + i] = sumGradients[gID] * activations[i];                            \n" \
"       //printf(\"Sum Gradient: %f    Activation: %f    Result:    %f\", sumGradients[gID], activations[i], weightGradients[start + i]);\n" \
"   }                                                                                               \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcGradientDZ_DB = "\n" \
"__kernel void calcGradientDZ_DB(__global float* sumGradients, __global float* biasGradients) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"                                                                                                   \n" \
"   biasGradients[gID] = sumGradients[gID];                                                         \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";

const char* calcGradientDZ_DA = "\n" \
"__kernel void calcGradientDZ_DA(__global float* sumGradients, __global float* weights, uint weightRows, uint weightCols, __global float* activationGradients) {  \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   uint gID = get_global_id(0);                                                                    \n" \
"                                                                                                   \n" \
"   float weightColSum = 0;                                                                         \n" \
"                                                                                                   \n" \
"   int sumRow = 0;                                                                                 \n" \
"   for (int i = gID; i < weightRows * weightCols; i += weightCols) {                               \n" \
"	    weightColSum += weights[i] * sumGradients[sumRow];                                          \n" \
"       sumRow++;                                                                                   \n" \
"   }                                                                                               \n" \
"                                                                                                   \n" \
"   activationGradients[gID] = weightColSum;                                                        \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";
























int NumberOfOutputKernels = 1;

int NumberOfCostKernels = 1;

int NumberOfSpecificGradientDC_DAKernels = 1;
int NumberOfSpecificGradientDA_DZKernels = 1;
int NumberOfGeneralGradientKernels = 3;

int NumberOfKernels = NumberOfOutputKernels + NumberOfCostKernels + NumberOfSpecificGradientDC_DAKernels + NumberOfSpecificGradientDA_DZKernels + NumberOfGeneralGradientKernels;

const char** kernels;

const char** neuralNetworkOutputKernelNames;

const char** neuralNetworkCostKernelNames;

const char** neuralNetworkSpecificGradientDC_DANames;
const char** neuralNetworkSpecificGradientDA_DZNames;

const char* neuralNetworkGeneralGradientDZ_DWName;
const char* neuralNetworkGeneralGradientDZ_DBName;
const char* neuralNetworkGeneralGradientDZ_DAName;

int initializeKernels() {
	kernels = (const char**)malloc(sizeof(const char*) * NumberOfKernels);
	neuralNetworkOutputKernelNames = (const char**)malloc(sizeof(const char*) * NumberOfOutputKernels);
	neuralNetworkCostKernelNames = (const char**)malloc(sizeof(const char*) * NumberOfCostKernels);
	neuralNetworkSpecificGradientDC_DANames = (const char**)malloc(sizeof(const char*) * NumberOfSpecificGradientDC_DAKernels);
	neuralNetworkSpecificGradientDA_DZNames = (const char**)malloc(sizeof(const char*) * NumberOfSpecificGradientDA_DZKernels);

	kernels[0] = calcOutSigmoid;
	kernels[1] = calcSquaredErrorCost;
	kernels[2] = calcMeanSquaredCostGradientDC_DA;
	kernels[3] = calcSigmoidGradientDA_DZ;
	kernels[4] = calcGradientDZ_DW;
	kernels[5] = calcGradientDZ_DB;
	kernels[6] = calcGradientDZ_DA;

	neuralNetworkOutputKernelNames[0] = "calcOutSigmoid";

	neuralNetworkCostKernelNames[0] = "calcSquaredErrorCost";

	neuralNetworkSpecificGradientDC_DANames = (const char**)malloc(sizeof(const char*) * NumberOfSpecificGradientDC_DAKernels);
	neuralNetworkSpecificGradientDC_DANames[0] = "calcMeanSquaredCostGradientDC_DA";

	neuralNetworkSpecificGradientDA_DZNames = (const char**)malloc(sizeof(const char*) * NumberOfSpecificGradientDA_DZKernels);
	neuralNetworkSpecificGradientDA_DZNames[0] = "calcSigmoidGradientDA_DZ";

	neuralNetworkGeneralGradientDZ_DWName = "calcGradientDZ_DW";
	neuralNetworkGeneralGradientDZ_DBName = "calcGradientDZ_DB";
	neuralNetworkGeneralGradientDZ_DAName = "calcGradientDZ_DA";

	return 0;
}

int releaseKernels() {
	for (int i = 0; i < NumberOfKernels; i++) {
		free((void*)(kernels[i]));
		free((void*)(neuralNetworkOutputKernelNames[i]));
	}

	free(kernels);
	free(neuralNetworkOutputKernelNames);
	//free(neuralNetworkGradientKernelNames);
	free(neuralNetworkCostKernelNames);

	return 0;
}
