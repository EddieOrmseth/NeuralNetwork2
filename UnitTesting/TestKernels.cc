#include "TestKernels.h"
#include "pch.h"

const char* vectorAdd = "\n" \
"__kernel void vectorAdd(__global const int* A, __global const int* B, __global int* C) {           \n" \
"   // Get the index of the current element to be processed                                         \n" \
"   int i = get_global_id(0);                                                                       \n" \
"                                                                                                   \n" \
"   // Do the operation                                                                             \n" \
"   int result = A[i] + B[i];                                                                       \n" \
"   printf(\"Global ID: %i    Result: %i\\n\", i, result);                                          \n" \
"   C[i] = result;                                                                                  \n" \
"                                                                                                   \n" \
"}                                                                                                  \n" \
"\n";