#include "pch.h"
#include "CppUnitTest.h"
#include <iostream>
#include <chrono>
#include "Windows.h"
#include "../NeuralNetwork2/OpenCLStuff.h"
#include "UnitTests.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

std::chrono::high_resolution_clock::time_point testStart = std::chrono::high_resolution_clock::now();

void beginTesting() {
	FreeConsole();
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	std::cout << "Beginning Test" << std::endl << std::endl;
	testStart = std::chrono::high_resolution_clock::now();
}

void beginTesting(const wchar_t* methodName) {
	FreeConsole();
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	std::wcout << "Beginning Test: " << methodName << std::endl << std::endl;
	testStart = std::chrono::high_resolution_clock::now();
}

void finishTesting() {
	std::chrono::high_resolution_clock::time_point testEnd = std::chrono::high_resolution_clock::now();
	long long time = std::chrono::duration_cast<std::chrono::milliseconds>(testEnd - testStart).count();
	std::cout << "\nTest Complete in " << std::to_string(time) << " milliseconds" << std::endl;
	std::cin.ignore();
}

namespace UnitTesting {

	TEST_CLASS(OpenCLSetupTesting) {
public:

	TEST_METHOD(DiscoverDevices) {
		beginTesting(__GetTestMethodInfo_DiscoverDevices()->metadata->methodName);
		printOpenCLDevices();
		finishTesting();
	}

	TEST_METHOD(PrintOnEachDevice) {
		beginTesting(__GetTestMethodInfo_PrintOnEachDevice()->metadata->methodName);
		PrintOnEachDeviceFunc();
		finishTesting();
	}

	TEST_METHOD(AddIntsFromBuffer) {
		beginTesting(__GetTestMethodInfo_AddIntsFromBuffer()->metadata->methodName);
		AddIntsFromBufferFunc();
		finishTesting();
	}

	TEST_METHOD(AddVectorsFromBuffer) {
		beginTesting(__GetTestMethodInfo_AddVectorsFromBuffer()->metadata->methodName);
		AddVectorsFromBufferFunc();
		finishTesting();
	}

	};

	TEST_CLASS(NeuralNetwokKernelUnitTests) {
public:
	TEST_METHOD(CalcOutSigmoidKernel) {
		beginTesting(__GetTestMethodInfo_CalcOutSigmoidKernel()->metadata->methodName);
		TestCalcOutSigmoidFunc();
		finishTesting();
	}

	TEST_METHOD(CalcSquaredErorrKernel) {
		beginTesting(__GetTestMethodInfo_CalcSquaredErorrKernel()->metadata->methodName);
		TestCalcSquaredErorrKernelFunc();
		finishTesting();
	}

	TEST_METHOD(CalcMeanSquaredCostGradientDC_DAKernel) {
		beginTesting(__GetTestMethodInfo_CalcMeanSquaredCostGradientDC_DAKernel()->metadata->methodName);
		TestCalcMeanSquaredCostGradientDC_DAKernelFunc();
		finishTesting();
	}

	TEST_METHOD(CalcSigmoidGradientDA_DZKernel) {
		beginTesting(__GetTestMethodInfo_CalcSigmoidGradientDA_DZKernel()->metadata->methodName);
		TestCalcSigmoidGradientDA_DZKernelFunc();
		finishTesting();
	}

	TEST_METHOD(CalcGradientDZ_DWKernel) {
		beginTesting(__GetTestMethodInfo_CalcGradientDZ_DWKernel()->metadata->methodName);
		TestCalcGradientDZ_DWKernelFunc();
		finishTesting();
	}

	TEST_METHOD(CalcGradientDZ_DBKernel) {
		beginTesting(__GetTestMethodInfo_CalcGradientDZ_DBKernel()->metadata->methodName);
		TestCalcGradientDZ_DBKernelFunc();
		finishTesting();
	}

	TEST_METHOD(CalcGradientDZ_DAKernel) {
		beginTesting(__GetTestMethodInfo_CalcGradientDZ_DAKernel()->metadata->methodName);
		TestCalcGradientDZ_DAKernelFunc();
		finishTesting();
	}

	};

	TEST_CLASS(NeuralNetworkTests) {
		TEST_METHOD(BasicNeuralNetworkOutput) {
			beginTesting(__GetTestMethodInfo_BasicNeuralNetworkOutput()->metadata->methodName);
			TestNeuralNetworkOutputFunc();
			finishTesting();
		}
	};



}
