#include "OpenCLData.h"
#include <string.h>

OpenCLBuffer* createOpenCLBufferHp(int entries, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result) {
	OpenCLBuffer* buffer = (OpenCLBuffer*)malloc(sizeof(OpenCLBuffer));
	int subresult = 0;

	buffer->bytesPerLength = bytesPerEntry;
	buffer->bufferLength = entries;
	buffer->bufferRawSize = entries * bytesPerEntry;
	buffer->buffer = clCreateBuffer(openCLInfo.context, memFlags, buffer->bufferRawSize, NULL, &subresult);
	*result |= subresult;

	buffer->hostData = malloc(buffer->bufferRawSize);

	return buffer;
}

OpenCLBuffer createOpenCLBufferStk(int entries, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result) {
	OpenCLBuffer buffer;
	int subresult = 0;

	buffer.bytesPerLength = bytesPerEntry;
	buffer.bufferLength = entries;
	buffer.bufferRawSize = entries * bytesPerEntry;
	buffer.buffer = clCreateBuffer(openCLInfo.context, memFlags, buffer.bufferRawSize, NULL, result);
	*result |= subresult;

	buffer.hostData = malloc(buffer.bufferRawSize);

	return buffer;
}

OpenCLMatrix* createOpenCLMatrixHp(int rows, int cols, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result) {
	OpenCLMatrix* matrix = (OpenCLMatrix*)malloc(sizeof(OpenCLMatrix));
	int subresult = 0;

	matrix->rows = rows;
	matrix->cols = cols;
	matrix->data = createOpenCLBufferStk(rows * cols, bytesPerEntry, openCLInfo, memFlags, result);
	*result |= subresult;

	return matrix;
}

OpenCLMatrix createOpenCLMatrixStk(int rows, int cols, int bytesPerEntry, OpenCLInfo openCLInfo, cl_mem_flags memFlags, int* result) {
	OpenCLMatrix matrix;
	int subresult = 0;

	matrix.rows = rows;
	matrix.cols = cols;
	matrix.data = createOpenCLBufferStk(rows * cols, bytesPerEntry, openCLInfo, memFlags, result);
	*result |= subresult;

	return matrix;
}

int readData(OpenCLInfo openCLInfo, OpenCLBuffer* buffer) {
	int result = 0;
	result |= clEnqueueReadBuffer(openCLInfo.commandQueue, buffer->buffer, CL_TRUE, 0, buffer->bufferRawSize, buffer->hostData, 0, 0, 0);
	result |= clFinish(openCLInfo.commandQueue);
	return result;
}

int writeData(OpenCLInfo openCLInfo, OpenCLBuffer* buffer) {
	int result = 0;
	result |= clEnqueueWriteBuffer(openCLInfo.commandQueue, buffer->buffer, CL_TRUE, 0, buffer->bufferRawSize, buffer->hostData, 0, 0, 0);
	result |= clFinish(openCLInfo.commandQueue);
	return result;
}

void printBuffer(OpenCLBuffer* buffer) {
	float* data = (float*)buffer->hostData;
	for (int i = 0; i < buffer->bufferLength; i++) {
		printf("%f  ", data[i]);
	}
	printf("\n");
}

void printBufferAsMatrix(OpenCLMatrix* matrix) {
	printBufferAsMatrix(&matrix->data, matrix->rows, matrix->cols);
}

void printBufferAsMatrix(OpenCLBuffer* buffer, int rows, int cols) {
	float* data = (float*)buffer->hostData;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			printf("%f  ", ((float*)buffer->hostData)[r * cols + c]);
		}
		printf("\n");
	}
}
