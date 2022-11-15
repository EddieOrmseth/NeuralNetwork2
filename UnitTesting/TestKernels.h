#pragma once

const char* printKernel = "\n" \
"__kernel void printKernel(/*__global char* string*/)                            \n" \
"{                                                                          \n" \
"   //char* string2 = string;                                               \n" \
"   //printf(\"Hello From \"); printf(string2); printf(\"\\n\");            \n" \
"   printf(\"\\t\\tHello!\\n\");                                                     \n" \
"}                                                                          \n" \
"\n";


const char* addOneInt = "\n" \
"__kernel void addOneInt(__global const int* input1, __global const int* input2, __global int* output)            \n" \
"{					    																			 \n" \
"	output[0] = input1[0] + input2[0];															     \n" \
"}		 																	  					    \n" \
"\n";


//const char* vectorAdd = "\n" \
//"__kernel void vectorAdd(__global const int* A, __global const int* B, __global int* C) {           \n" \
//"   // Get the index of the current element to be processed                                         \n" \
//"   int i = get_global_id(0);                                                                       \n" \
//"                                                                                                   \n" \
//"   // Do the operation                                                                             \n" \
//"   int result = A[i] + B[i];                                                                       \n" \
//"   printf(\"Global ID: %i    Result: %i\\n\", i, result);                                          \n" \
//"   C[i] = result;                                                                                  \n" \
//"                                                                                                   \n" \
//"}                                                                                                  \n" \
//"\n";

extern const char* vectorAdd;
