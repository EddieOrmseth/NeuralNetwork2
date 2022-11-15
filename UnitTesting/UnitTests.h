#pragma once
#include <iostream>
#include "NeuralNetwork.h"

int PrintOnEachDeviceFunc();

int runOnAllDevices(const char* kernelPtr, char* kernelName);
int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result));
int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), int workDim, size_t globalWorkOffset, size_t globalWorkSize, size_t localWorkSize);
int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), void (*checkResults)(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device));
int runOnAllDevices(const char* kernelPtr, char* kernelName, OpenCLBuffer** (*configureBuffers)(OpenCLInfo* openCLInfo, int* numBuffers, int* result), void (*checkResults)(OpenCLInfo* openCLInfo, OpenCLBuffer** buffers, int platform, int device), int workDim, size_t globalWorkOffset, size_t globalWorkSize, size_t localWorkSize);

int runOnAllDevices(int (*function)(int platformNum, int deviceNum));

int AddIntsFromBufferFunc();

int AddVectorsFromBufferFunc();

int TestCalcOutSigmoidFunc();

int TestCalcSquaredErorrKernelFunc();

int TestCalcMeanSquaredCostGradientDC_DAKernelFunc();

int TestCalcSigmoidGradientDA_DZKernelFunc();

int TestCalcGradientDZ_DWKernelFunc();

int TestCalcGradientDZ_DBKernelFunc();

int TestCalcGradientDZ_DAKernelFunc();

int TestNeuralNetworkOutputFunc();