#include "pch.h"
#include "UnitTests.h"
#include <Windows.h>
#include "TestKernels.h"
#include "BackPropNeuralNetwork.h"

int PrintOnEachDeviceFunc() {
    int result = 0;

    cl_uint numPlatforms = 0;
    cl_platform_id* platforms;
    result = clGetPlatformIDs(NULL, NULL, &numPlatforms);
    if (result != CL_SUCCESS) {
        printf("Error: Failed to create a platform group\n");
        return -1;
    }
    else {
        //printf("Number of Platforms Found: %i\n", numPlatforms);
        platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
        result = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (result != CL_SUCCESS) {
            printf("Error: Failed to create a platform group\n");
            return -1;
        }
    }

    cl_uint numDevices = 0;
    cl_device_id* devices;

    char* name;
    size_t paramSize = 0;

    for (int p = 0; p < numPlatforms; p++) {

        result = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, NULL, NULL, &paramSize);
        if (result != CL_SUCCESS) {
            printf("Error: Unable to retrieve ParamSize from Platform %i", p);
        }
        else {
            name = (char*)malloc(sizeof(char) * paramSize);
            result = clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, paramSize, name, NULL);
            if (result != CL_SUCCESS) {
                printf("Error: Unable to retrieve Name from Platform %i", p);
            }
            else {
                printf("Platform %i: ", p); printf(name); printf("\n");
            }

            free(name);
        }

        result = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, NULL, NULL, &numDevices);
        if (result != CL_SUCCESS) {
            printf("Error: Unable to retrieve number of Devices from Platform %i", p);
            continue;
        }
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        result = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        if (result != CL_SUCCESS) {
            printf("Error: Unable to retrieve Devices from Platform %i", p);
            continue;
        }

        for (int d = 0; d < numDevices; d++) {
            result = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, NULL, NULL, &paramSize);
            if (result != CL_SUCCESS) {
                printf("Error: Unable to retrieve ParamSize from Device %i", d);
                continue;
            }
            name = (char*)malloc(sizeof(char) * paramSize);
            result = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, paramSize, name, NULL);
            if (result != CL_SUCCESS) {
                printf("Error: Unable to retrieve Name from Device %i", d);
                continue;
            }

            printf("\t\tDevice %i: ", d); printf(name); printf("\n");

            // Print with Kernel
            OpenCLInfo* openCLInfo = OpenCLSetup(p, d, &printKernel, 1);

            cl_kernel toRun = clCreateKernel(openCLInfo->program, "printKernel", &result);

            if (result != 0) {
                printf("Error: Failed to get kernel with name \"printKernel\"");
                continue;
            }

            size_t one = 1;
            clEnqueueNDRangeKernel(openCLInfo->commandQueue, toRun, 1, &one, &one, &one, 0, 0, 0);

            clFinish(openCLInfo->commandQueue);

            free(name);
        }

        free(devices);
    }

    free(platforms);

    return 0;
}

int runOnAllDevices(const char* kernelPtr, char* kernelName) {
    return runOnAllDevices(kernelPtr, kernelName, nullptr);
}

int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result)) {
    return runOnAllDevices(kernelPtr, kernelName, configureBuffers, nullptr, 1, 0, 1, 1);
}

int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), int workDim, size_t globalWorkOffset, size_t globalWorkSize, size_t localWorkSize) {
    return runOnAllDevices(kernelPtr, kernelName, configureBuffers, nullptr);
}

int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), void (*checkResults)(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device)) {
    return runOnAllDevices(kernelPtr, kernelName, configureBuffers, checkResults, 1, 0, 1, 1);
}

int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), void (*checkResults)(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device), int workDim, size_t globalWorkOffset, size_t globalWorkSize, size_t localWorkSize) {
    int numPlatforms = getNumPlatforms();
    for (int p = 0; p < numPlatforms; p++) {
        int numDevices = getNumDevices(p);
        printf("Platform %i:\n", p);
        for (int d = 0; d < numDevices; d++) {
            printf("\tDevice %i:\n", d);
            OpenCLInfo* openCLInfo = OpenCLSetup(p, d, &kernelPtr, 1);
            int result = 0;
            cl_kernel kernel = clCreateKernel(openCLInfo->program, kernelName, &result);
            if (result != CL_SUCCESS) {
                printf("\t\tError: Unable to create kernel\n");
                free(openCLInfo);
                continue;
            }

            int numBuffers = 0;
            OpenCLBuffer** args = nullptr;
            if (configureBuffers != nullptr) {
                result = 0;
                args = (*configureBuffers)(openCLInfo, &numBuffers, &result);
                if (result != 0) {
                    printf("\t\tError: Failed to configure buffers\n");
                    free(openCLInfo);
                    continue;
                }

                for (int i = 0; i < numBuffers; i++) {
                    result = clSetKernelArg(kernel, i, sizeof(args[i]->buffer), &args[i]->buffer);
                    if (result != CL_SUCCESS) {
                        printf("\t\tError: Unable to set kernel argument %i\n", i);
                        continue;
                    }
                }
            }

            result = clEnqueueNDRangeKernel(openCLInfo->commandQueue, kernel, workDim, &globalWorkOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);
            if (result != CL_SUCCESS) {
                printf("\t\tError: Unable to run kernel\n");
            }

            result = clFinish(openCLInfo->commandQueue);
            if (result != CL_SUCCESS) {
                printf("\t\tError: Unable to finsih CommandQueue\n");
            }

            if (checkResults != nullptr) {
                (*checkResults)(openCLInfo, args, p, d);
            }

            if (numBuffers > 0) {
                for (int i = 0; i < numBuffers; i++) {
                    free(args[i]);
                }
                free(args);
            }

            free(openCLInfo);
        }
    }
    
    return 0;
}

int runOnAllDevices(int (*function)(int platformNum, int deviceNum)) {
    int result = 0;
    int subResult = 0;

    int numPlatforms = getNumPlatforms();
    for (int p = 0; p < numPlatforms; p++) {
        int numDevices = getNumDevices(p);
        printf("Platform %i:\n", p);
        for (int d = 0; d < numDevices; d++) {
            printf("\tDevice %i:\n", d);

            subResult = (*function)(p, d);
            result |= subResult;

            if (subResult == 0) {
                printf("\t\tTest Passed!\n\n");
            }
            else {
                printf("\t\tTest Failed. Error Code: %i\n\n", subResult);
            }

        }
    }

    return result;
}

OpenCLBuffer** AddIntsFromBufferFuncConfigureBuffers(OpenCLInfo* openCLInfo, int* numBuffers, int* result) {
    *numBuffers = 3;
    int subresult = 0;

    OpenCLBuffer** buffers = (OpenCLBuffer**)malloc(sizeof(OpenCLBuffer*) * *numBuffers);
    buffers[0] = createOpenCLBufferHp(1, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);
    buffers[1] = createOpenCLBufferHp(1, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);
    buffers[2] = createOpenCLBufferHp(1, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);

    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to Create Buffer\n");
    }

    int toWrite = 1;
    subresult |= clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[0]->buffer, CL_TRUE, 0, buffers[0]->bufferRawSize, &toWrite, 0, 0, 0);
    subresult |= clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[1]->buffer, CL_TRUE, 0, buffers[1]->bufferRawSize, &toWrite, 0, 0, 0);
    toWrite = 0;
    subresult |= clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, &toWrite, 0, 0, 0);

    clFinish(openCLInfo->commandQueue);

    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to Write to Buffers\n");
    }

    *result |= subresult;

    return buffers;
}

void AddIntsFromBufferFuncResultCheck(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device) {
    int read = 0;
    clEnqueueReadBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, &read, 0, 0, 0);

    clFinish(openCLInfo->commandQueue);

    if (read == 2) {
        printf("\t\tSuccessfully Added! Test Passed!\n");
    }
    else {
        printf("\t\tFailed to add correctly! Test Failed.\n");
    }

    read = 0;
    clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, &read, 0, 0, 0);

    clFinish(openCLInfo->commandQueue);
}

int AddIntsFromBufferFunc() {
    return runOnAllDevices(addOneInt, "addOneInt", AddIntsFromBufferFuncConfigureBuffers, AddIntsFromBufferFuncResultCheck);
}


OpenCLBuffer** AddVectorsFromBufferFuncConfigureBuffers(OpenCLInfo* openCLInfo, int* numBuffers, int* result) {
    *numBuffers = 3;
    int subresult = 0;
    int vectorSize = 10;

    OpenCLBuffer** buffers = (OpenCLBuffer**)malloc(sizeof(OpenCLBuffer*) * *numBuffers);
    buffers[0] = createOpenCLBufferHp(vectorSize, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);
    buffers[1] = createOpenCLBufferHp(vectorSize, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);
    buffers[2] = createOpenCLBufferHp(vectorSize, sizeof(int), *openCLInfo, CL_MEM_READ_WRITE, &subresult);

    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to Create Buffer\n");
    }

    int* vecA = (int*)malloc(sizeof(int) * vectorSize);
    int* vecB = (int*)malloc(sizeof(int) * vectorSize);

    for (int i = 0; i < vectorSize; i++) {
        vecA[i] = i;
        vecB[i] = vectorSize - (i);
    }

    subresult = 0;
    subresult |= clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[0]->buffer, CL_TRUE, 0, buffers[0]->bufferRawSize, vecA, 0, 0, 0);
    subresult |= clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[1]->buffer, CL_TRUE, 0, buffers[1]->bufferRawSize, vecB, 0, 0, 0);

    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to Write to Buffers\n");
    }

    clFinish(openCLInfo->commandQueue);

    *result |= subresult;

    return buffers;
}

void AddVectorsFromBufferFuncResultCheck(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device) {
    int vectorSize = 10;
    int* vecC = (int*)malloc(sizeof(int) * vectorSize);

    int subresult = 0;

    subresult = clEnqueueReadBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, vecC, 0, 0, 0);
    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to read buffer in ResultCheck\n");
    }

    clFinish(openCLInfo->commandQueue);

    bool succeeded = true;
    for (int i = 0; i < vectorSize; i++) {
        if (vecC[i] != 10) {
            succeeded = false;
        }
        printf("%i: %i  ", i, vecC[i]);
    }
    printf("\n");

    if (succeeded) {
        printf("\t\tSuccessfully Added! Test Passed!\n");
    }
    else {
        printf("\t\tFailed to add correctly! Test Failed.\n");
    }

    memset(vecC, 0, sizeof(int) * vectorSize);
    clEnqueueWriteBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, vecC, 0, 0, 0);
    clFinish(openCLInfo->commandQueue);

    subresult = clEnqueueReadBuffer(openCLInfo->commandQueue, buffers[2]->buffer, CL_TRUE, 0, buffers[2]->bufferRawSize, vecC, 0, 0, 0);
    if (subresult != CL_SUCCESS) {
        printf("\t\tError: Failed to read buffer in ResultCheck\n");
    }

    clFinish(openCLInfo->commandQueue);

    succeeded = true;
    for (int i = 0; i < vectorSize; i++) {
        if (vecC[i] != 10) {
            succeeded = false;
        }
        printf("%i: %i  ", i, vecC[i]);
    }
    printf("\n");
}

int AddVectorsFromBufferFunc() {
    return runOnAllDevices(vectorAdd, "vectorAdd", AddVectorsFromBufferFuncConfigureBuffers, AddVectorsFromBufferFuncResultCheck, 1, 0, 10, 1);
}

int DoTestCalcOutSigmoid(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);

    cl_int layer1Size = 3;
    cl_int layer2Size = 4;

    OpenCLBuffer* inputs = createOpenCLBufferHp(layer1Size, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLMatrix* weights = createOpenCLMatrixHp(layer2Size, layer1Size, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* biases = createOpenCLBufferHp(layer2Size, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    OpenCLBuffer* sums = createOpenCLBufferHp(layer2Size, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* results = createOpenCLBufferHp(layer2Size, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* writeInputs = (float*)inputs->hostData;
    for (int i = 0; i < layer1Size; i++) {
        writeInputs[i] = 1;
    }
    result |= writeData(openCLInfo, inputs);

    float* writeWeights = (float*)weights->data.hostData;
    for (int i = 0; i < layer1Size * layer2Size; i++) {
        writeWeights[i] = i + 1;
    }
    result |= writeData(openCLInfo, &weights->data);

    float* writeBiases = (float*)biases->hostData;
    for (int i = 0; i < layer2Size; i++) {
        writeBiases[i] = .5;
    }
    result |= writeData(openCLInfo, biases);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkOutputKernelNames[ActivationFunction::Sigmoid], &result);
    result |= clSetKernelArg(kernel, 0, sizeof(inputs->buffer), &inputs->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(cl_int), &layer1Size);
    result |= clSetKernelArg(kernel, 2, sizeof(weights->data.buffer), &weights->data.buffer);
    result |= clSetKernelArg(kernel, 3, sizeof(biases->buffer), &biases->buffer);
    result |= clSetKernelArg(kernel, 4, sizeof(sums->buffer), &sums->buffer);
    result |= clSetKernelArg(kernel, 5, sizeof(results->buffer), &results->buffer);

    size_t globalOffset = 0;
    size_t globalSize = layer2Size;
    size_t localSize = 1;
    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalSize, &localSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    readData(openCLInfo, inputs);
    readData(openCLInfo, &weights->data);
    readData(openCLInfo, biases);
    readData(openCLInfo, sums);
    readData(openCLInfo, results);

    printf("Inputs: \n");  printBuffer(inputs); printf("\n");
    printf("Weights: \n");  printBufferAsMatrix(&weights->data, weights->rows, weights->cols); printf("\n");
    printf("Biases: \n");  printBuffer(biases); printf("\n");
    printf("Sums: \n");  printBuffer(sums); printf("\n");
    printf("Results: \n");  printBuffer(results); printf("\n");

    float* readSums = (float*)sums->hostData;
    float* readResults = (float*)results->hostData;

    if (readSums[0] == 6.5 && readSums[1] == 15.5 && readSums[2] == 24.5 && readSums[3] == 33.5 &&
        readResults[0] == 0.998498797f && readResults[1] == 0.999999762f && readResults[2] == 1.00000000f && readResults[3] == 1.00000000f) {
        return result;
    }
    else {
        return 1;
    }

}

int TestCalcOutSigmoidFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcOutSigmoid);
}

int DoTestCalcSquaredErorrKernel(int platformNum, int deviceNum) {

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);

    int result = 0;

    int numEntries = 3;
    OpenCLBuffer* nnOutput = createOpenCLBufferHp(numEntries, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* target = createOpenCLBufferHp(numEntries, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* costs = createOpenCLBufferHp(numEntries, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* nnOutputData = (float*)nnOutput->hostData;
    for (int i = 0; i < numEntries; i++) {
        nnOutputData[i] = 1;
    }

    float* targetData = (float*)target->hostData;
    for (int i = 0; i < numEntries; i++) {
        targetData[i] = 3;
    }

    float* costsData = (float*)costs->hostData;

    result |= writeData(openCLInfo, nnOutput);
    result |= writeData(openCLInfo, target);

    cl_kernel costQuadKernel = clCreateKernel(openCLInfo.program, neuralNetworkCostKernelNames[0], &result);
    result |= clSetKernelArg(costQuadKernel, 0, sizeof(nnOutput->buffer), &nnOutput->buffer);
    result |= clSetKernelArg(costQuadKernel, 1, sizeof(target->buffer), &target->buffer);
    result |= clSetKernelArg(costQuadKernel, 2, sizeof(costs->buffer), &costs->buffer);

    size_t globalOffset = 0;
    size_t globalWorkSize = numEntries;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, costQuadKernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, nnOutput);
    result |= readData(openCLInfo, target);
    result |= readData(openCLInfo, costs);

    printf("Input: \n");printBuffer(nnOutput); printf("\n");
    printf("Target: \n"); printBuffer(target); printf("\n");
    printf("Result: \n"); printBuffer(costs); printf("\n");

    if (costsData[0] == 4.0 && costsData[1] == 4.0 && costsData[2] == 4.0) {
        return result;
    }
    else {
        return 1;
    }

}

int TestCalcSquaredErorrKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcSquaredErorrKernel);
}

int DoTestCalcMeanSquaredCostGradientDC_DAKernel(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);
    int layerSize = 10;

    OpenCLBuffer* nnOutput = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    cl_float outputSizeReciprocal = 1.0f / layerSize;
    OpenCLBuffer* target = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* costGradients = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* nnOutputHost = (float*)nnOutput->hostData;
    for (int i = 0; i < layerSize; i++) {
        nnOutputHost[i] = 1;
    }
    result |= writeData(openCLInfo, nnOutput);

    float* targetHost = (float*)target->hostData;
    for (int i = 0; i < layerSize; i++) {
        targetHost[i] = i + 1;
    }
    result |= writeData(openCLInfo, target);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkSpecificGradientDC_DANames[CostFunction::MeanSquared], &result);

    result |= clSetKernelArg(kernel, 0, sizeof(nnOutput->buffer), &nnOutput->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(outputSizeReciprocal), &outputSizeReciprocal);
    result |= clSetKernelArg(kernel, 2, sizeof(target->buffer), &target->buffer);
    result |= clSetKernelArg(kernel, 3, sizeof(costGradients->buffer), &costGradients->buffer);

    size_t globalOffset = 0;
    size_t globalWorkSize = layerSize;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, nnOutput);
    result |= readData(openCLInfo, target);
    result |= readData(openCLInfo, costGradients);

    printf("Reciprocal Size: %f\n", outputSizeReciprocal);
    printf("NN Output: \n"); printBuffer(nnOutput); printf("\n");
    printf("Target: \n"); printBuffer(target); printf("\n");
    printf("Cost Gradients: \n"); printBuffer(costGradients); printf("\n");

    float* costGradientsHost = (float*)costGradients->hostData;
    if (costGradientsHost[0] == 0.00000000f && costGradientsHost[1] == -0.200000003f && costGradientsHost[2] == -0.400000006f && costGradientsHost[3] == -0.600000024f &&
        costGradientsHost[4] == -0.800000012f && costGradientsHost[5] == -1.00000000f && costGradientsHost[6] == -1.20000005f && costGradientsHost[7] == -1.39999998f &&
        costGradientsHost[8] == -1.60000002f && costGradientsHost[9] == -1.80000007f) {
        return result;
    }
    else {
        return 1;
    }

}

int TestCalcMeanSquaredCostGradientDC_DAKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcMeanSquaredCostGradientDC_DAKernel);
}

int DoTestCalcSigmoidGradientDA_DZKernel(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);
    int layerSize = 4;

    OpenCLBuffer* activationGradients = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* sums = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* sumGradients = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* activationGradientsHost = (float*)activationGradients->hostData;
    float* sumsHost = (float*)sums->hostData;
    for (int i = 0; i < layerSize; i++) {
        activationGradientsHost[i] = i + 1;
        sumsHost[i] = layerSize - i;
    }

    result |= writeData(openCLInfo, activationGradients);
    result |= writeData(openCLInfo, sums);
    result |= writeData(openCLInfo, sumGradients);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkSpecificGradientDA_DZNames[CostFunction::MeanSquared], &result);

    result |= clSetKernelArg(kernel, 0, sizeof(activationGradients->buffer), &activationGradients->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(sums->buffer), &sums->buffer);
    result |= clSetKernelArg(kernel, 2, sizeof(sumGradients->buffer), &sumGradients->buffer);

    size_t globalOffset = 0;
    size_t globalWorkSize = layerSize;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, activationGradients);
    result |= readData(openCLInfo, sums);
    result |= readData(openCLInfo, sumGradients);

    printf("Activation Gradients: \n"); printBuffer(activationGradients); printf("\n");
    printf("Sums: \n"); printBuffer(sums); printf("\n");
    printf("Sum Gradients: \n"); printBuffer(sumGradients); printf("\n");

    float* sumGradientsHost = (float*)sumGradients->hostData;
    if (sumGradientsHost[0] == 0.0176627077f && sumGradientsHost[1] == 0.0903533325f && sumGradientsHost[2] == 0.314980745f && sumGradientsHost[3] == 0.786447763f) {
        return result;
    }
    else {
        return 1;
    }

}

int TestCalcSigmoidGradientDA_DZKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcSigmoidGradientDA_DZKernel);
}

int DoTestCalcGradientDZ_DWKernel(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);
    cl_uint weightsRows = 4;
    cl_uint weightsCols = 5;

    OpenCLBuffer* sumGradients = createOpenCLBufferHp(weightsRows, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* activations = createOpenCLBufferHp(weightsCols, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLMatrix* weightGradients = createOpenCLMatrixHp(weightsRows, weightsCols, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* sumGradientsHost = (float*)sumGradients->hostData;
    for (int i = 0; i < weightsRows; i++) {
        sumGradientsHost[i] = i + 1;
    }
    result |= writeData(openCLInfo, sumGradients);

    float* activationsHost = (float*)activations->hostData;
    for (int i = 0; i < weightsCols; i++) {
        activationsHost[i] = weightsCols - i;
    }
    result |= writeData(openCLInfo, activations);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DWName, &result);

    result |= clSetKernelArg(kernel, 0, sizeof(sumGradients->buffer), &sumGradients->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(activations->buffer), &activations->buffer);
    result |= clSetKernelArg(kernel, 2, sizeof(weightGradients->data.buffer), &weightGradients->data.buffer);
    result |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &weightsCols);

    //printf("SumGradients: %i    Activations: %i    WeightGradients: %i X %i\n", sumGradients->bufferLength, activations->bufferLength, weightGradients->rows, weightGradients->cols);

    size_t globalOffset = 0;
    size_t globalWorkSize = weightsRows;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, sumGradients);
    result |= readData(openCLInfo, activations);
    result |= readData(openCLInfo, &weightGradients->data);

    printf("Sum Gradients: \n"); printBuffer(sumGradients); printf("\n");
    printf("Activations: \n"); printBuffer(activations); printf("\n");
    printf("Weight Gradients: \n"); printBufferAsMatrix(&weightGradients->data, weightGradients->rows, weightGradients->cols); printf("\n");

    float* weightGradientsHost = (float*)weightGradients->data.hostData;
    for (int r = 0; r < weightsRows; r++) {
        for (int c = 0; c < weightsCols; c++) {
            if (weightGradientsHost[r * weightsCols + c] != sumGradientsHost[r] * activationsHost[c]) {
                return 1;
            }
        }
    }
    
    return result;
}

int TestCalcGradientDZ_DWKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcGradientDZ_DWKernel);
}

int DoTestCalcGradientDZ_DBKernel(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);
    cl_uint layerSize = 5;

    OpenCLBuffer* sumGradients = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* biasGradients = createOpenCLBufferHp(layerSize, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* sumGradientsHost = (float*)sumGradients->hostData;
    for (int i = 0; i < layerSize; i++) {
        sumGradientsHost[i] = i + 1;
    }
    result |= writeData(openCLInfo, sumGradients);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DBName, &result);

    result |= clSetKernelArg(kernel, 0, sizeof(sumGradients->buffer), &sumGradients->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(biasGradients->buffer), &biasGradients->buffer);

    size_t globalOffset = 0;
    size_t globalWorkSize = layerSize;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, sumGradients);
    result |= readData(openCLInfo, biasGradients);

    printf("Sum Gradients: \n"); printBuffer(sumGradients); printf("\n");
    printf("Bias Gradients: \n"); printBuffer(biasGradients); printf("\n");

    float* biasGradientsHost = (float*)biasGradients->hostData;
    for (int i = 0; i < layerSize; i++) {
        if (biasGradientsHost[i] != i + 1) {
            return 1;
        }
    }

    return result;
}

int TestCalcGradientDZ_DBKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcGradientDZ_DBKernel);
}

int DoTestCalcGradientDZ_DAKernel(int platformNum, int deviceNum) {
    int result = 0;

    OpenCLInfo openCLInfo = *OpenCLSetup(platformNum, deviceNum, kernels, NumberOfKernels);
    cl_uint weightRows = 4;
    cl_uint weightCols = 5;

    OpenCLBuffer* sumGradients = createOpenCLBufferHp(weightRows, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLMatrix* weights = createOpenCLMatrixHp(weightRows, weightCols, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);
    OpenCLBuffer* activationGradients = createOpenCLBufferHp(weightCols, sizeof(float), openCLInfo, CL_MEM_READ_WRITE, &result);

    float* sumGradientsHost = (float*)sumGradients->hostData;
    for (int i = 0; i < weightRows; i++) {
        sumGradientsHost[i] = i + 1;
    }
    result |= writeData(openCLInfo, sumGradients);

    float* weightHost = (float*)weights->data.hostData;
    for (int i = 0; i < weightRows * weightCols; i++) {
        weightHost[i] = i + 1;
    }
    result |= writeData(openCLInfo, &weights->data);

    cl_kernel kernel = clCreateKernel(openCLInfo.program, neuralNetworkGeneralGradientDZ_DAName, &result);

    result |= clSetKernelArg(kernel, 0, sizeof(sumGradients->buffer), &sumGradients->buffer);
    result |= clSetKernelArg(kernel, 1, sizeof(weights->data.buffer), &weights->data.buffer);
    result |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &weightRows);
    result |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &weightCols);
    result |= clSetKernelArg(kernel, 4, sizeof(activationGradients->buffer), &activationGradients->buffer);

    size_t globalOffset = 0;
    size_t globalWorkSize = weightCols;
    size_t localWorkSize = 1;

    result |= clEnqueueNDRangeKernel(openCLInfo.commandQueue, kernel, 1, &globalOffset, &globalWorkSize, &localWorkSize, 0, 0, 0);

    result |= clFinish(openCLInfo.commandQueue);

    result |= readData(openCLInfo, sumGradients);
    result |= readData(openCLInfo, &weights->data);
    result |= readData(openCLInfo, activationGradients);

    printf("Sum Gradients: \n"); printBuffer(sumGradients); printf("\n");
    printf("Weights: \n"); printBufferAsMatrix(&weights->data, weightRows, weightCols); printf("\n");
    printf("Activation Gradients: \n"); printBuffer(activationGradients); printf("\n");

    float* activationGradientsHost = (float*)activationGradients->hostData;
    if (activationGradientsHost[0] == 110.0f && activationGradientsHost[1] == 120.0f && activationGradientsHost[2] == 130.0f &&
        activationGradientsHost[3] == 140.0f && activationGradientsHost[4] == 150.0f) {
        return result;
    }
    else {
        return 1;
    }

}

int TestCalcGradientDZ_DAKernelFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestCalcGradientDZ_DAKernel);
}


int DoTestNeuralNetworkOutput(int platformNum, int deviceNum) {
    int result = 0;

    initializeKernels();

    OpenCLInfo* openCLInfo = OpenCLSetup(1, 0, kernels, NumberOfKernels);
    int numLayers = 3;
    int* layers = (int*)malloc(sizeof(int) * numLayers);
    layers[0] = 3;
    layers[1] = 5;
    layers[2] = 3;
    NeuralNetwork* network = createNeuralNetwork(layers, numLayers, *openCLInfo);


    float* input = (float*)network->layers[0].hostData;
    for (int i = 0; i < network->layers[0].bufferLength; i++) {
        input[i] = 1;
    }

    float* weights1 = (float*)network->weights[0].data.hostData;
    for (int i = 0; i < network->weights[0].data.bufferLength; i++) {
        weights1[i] = i + 1;
    }

    float* weights2 = (float*)network->weights[1].data.hostData;
    for (int i = 0; i < network->weights[1].data.bufferLength; i++) {
        weights2[i] = i + 1;
    }

    float* biases1 = (float*)network->biases[0].hostData;
    for (int i = 0; i < network->biases[0].bufferLength; i++) {
        biases1[i] = .5;
    }

    float* biases2 = (float*)network->biases[1].hostData;
    for (int i = 0; i < network->biases[1].bufferLength; i++) {
        biases2[i] = .5;
    }


    result |= writeAll(network);
    result |= calcNeuralNetworkOutput(network);
    result |= readAll(network);

    printNeuralNetwork(network);

    float* outputLayer = (float*)network->layers[network->numLayers - 1].hostData;

    if (outputLayer[0] == 0.999999762f && outputLayer[1] == 1.00000000f && outputLayer[2] == 1.00000000f) {
        return result;
    }
    else {
        return 1;
    }
}

int TestNeuralNetworkOutputFunc() {
    initializeKernels();
    return runOnAllDevices(DoTestNeuralNetworkOutput);
}