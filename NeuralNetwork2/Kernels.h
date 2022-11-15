#pragma once

extern int NumberOfKernels;

extern const char** kernels;

extern const char** neuralNetworkOutputKernelNames;

extern const char** neuralNetworkCostKernelNames;

extern const char** neuralNetworkSpecificGradientDC_DANames;
extern const char** neuralNetworkSpecificGradientDA_DZNames;

extern const char* neuralNetworkGeneralGradientDZ_DWName;
extern const char* neuralNetworkGeneralGradientDZ_DBName;
extern const char* neuralNetworkGeneralGradientDZ_DAName;

int initializeKernels();

int releaseKernels();
