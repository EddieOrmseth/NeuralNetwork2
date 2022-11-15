#include "OpenCLStuff.h"

#define __NO_STD_VECTOR // Use cl::vector instead of STD version
#define __CL_ENABLE_EXCEPTIONS // Error handling
#define CL_HPP_ENABLE_EXCEPTIONS // Error handling

OpenCLInfo* OpenCLSetup(int platformNum, int deviceNum, const char** kernels, int kernelCount) {
    OpenCLInfo* openCLInfo = (OpenCLInfo*)malloc(sizeof(OpenCLInfo));
    int result = 0;

    cl_uint numPlatforms = 0;
    cl_platform_id* platforms;
    result = clGetPlatformIDs(NULL, NULL, &numPlatforms);
    if (result != CL_SUCCESS) {
        printf("Error: Failed to create a platform group\n");
        return nullptr;
    }
    else {
        //printf("Number of Platforms Found: %i\n", numPlatforms);
        platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
        result = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (result != CL_SUCCESS) {
            printf("Error: Failed to create a platform group\n");
            return nullptr;
        }
    }
    cl_platform_id platform = platforms[platformNum];

    cl_uint numDevices = 0;
    cl_device_id* devices;
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, NULL, NULL, &numDevices);
    if (result != CL_SUCCESS) {
        printf("Error: Unable to retrieve number of Devices from Platform %i", platformNum);
        return nullptr;
    }
    else {
        devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
        result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
        if (result != CL_SUCCESS) {
            printf("Error: Unable to retrieve Devices from Platform %i", platformNum);
            return nullptr;
        }
    }
    cl_device_id device = devices[deviceNum];



    cl_context context = clCreateContext(0, 1, devices, NULL, NULL, &result);
    if (result != CL_SUCCESS) { // !context
        printf("Error: Failed to create a compute context!\n");
        return nullptr;
    }

    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device, 0, &result);
    if (result != CL_SUCCESS) { // !commandQueue
        printf("Error: Failed to create a command commands!\n");
        return nullptr;
    }

    cl_program program = clCreateProgramWithSource(context, kernelCount, kernels, NULL, &result);
    if (result != CL_SUCCESS) { // !program
        printf("Error: Failed to create compute program!\n");
        return nullptr;
    }
    result = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (result != CL_SUCCESS) {
        printf("Error: Failed to build program!\n");
        cl_program_build_info info;
        size_t paramSize = 0;
        result = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, NULL, NULL, &paramSize);
        if (result != CL_SUCCESS) {
            printf("Error: Failed to retrieve build error message length!\n");
            return nullptr;
        }
        char* message = (char*)malloc(paramSize);
        result = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, paramSize, message, NULL);
        printf(message); printf("\n");
        return nullptr;
    }

    openCLInfo->platform = platform;
    openCLInfo->device = device;
    openCLInfo->context = context;
    openCLInfo->program = program;
    openCLInfo->commandQueue = commandQueue;

    return openCLInfo;
}

int printOpenCLDevices() {
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
            free(name);
        }

        free(devices);
    }

    free(platforms);

    return 0;
}

int getNumPlatforms() {
    int result = 0;

    cl_uint numPlatforms = 0;
    cl_platform_id* platforms;
    result = clGetPlatformIDs(NULL, NULL, &numPlatforms);
    if (result != CL_SUCCESS) {
        printf("Error: Failed to create a platform group\n");
        return -1;
    }

    return numPlatforms;
}

int getNumDevices(int platformNum) {
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
    cl_platform_id platform = platforms[platformNum];

    cl_uint numDevices = 0;
    cl_device_id* devices;
    result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, NULL, NULL, &numDevices);
    if (result != CL_SUCCESS) {
        printf("Error: Unable to retrieve number of Devices from Platform %i", platformNum);
        return -1;
    }

    return numDevices;
}

const char* getErrorMessage(cl_int error)
{
    switch (error) {
        // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}