#pragma once
#include <stdlib.h>
#include <stdio.h>
#include "CL\cl.h"

struct OpenCLInfo {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_program program;
	cl_command_queue commandQueue;
};

OpenCLInfo* OpenCLSetup(int platformNum, int deviceNum, const char** kernels, int kernelCount);

int printOpenCLDevices();

int getNumPlatforms();
int getNumDevices(int platformNum);


const char* getErrorMessage(cl_int errorCode);