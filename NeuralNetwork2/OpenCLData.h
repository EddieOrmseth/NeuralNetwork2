#pragma once
#include "OpenCLStuff.h"

struct OpenCLBuffer {
	cl_int bytesPerLength;
	cl_int bufferLength;

	cl_int bufferRawSize;

	cl_mem buffer;
	void* hostData;
};

struct OpenCLMatrix {
	cl_int rows;
	cl_int cols;

	OpenCLBuffer data;
};

OpenCLBuffer* createOpenCLBufferHp(int entries, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result);
OpenCLBuffer createOpenCLBufferStk(int entries, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result);

OpenCLMatrix* createOpenCLMatrixHp(int rows, int cols, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result);
OpenCLMatrix createOpenCLMatrixStk(int rows, int cols, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result);

int readData(OpenCLInfo openCLInfo, OpenCLBuffer* buffer);
int writeData(OpenCLInfo openCLInfo, OpenCLBuffer* buffer);

void printBuffer(OpenCLBuffer* buffer);
void printBufferAsMatrix(OpenCLMatrix* buffer);
void printBufferAsMatrix(OpenCLBuffer* buffer, int rows, int cols);
