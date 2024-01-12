#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <string.h>
#include <math.h>
#include <igraph.h>

//#define NDEBUG

#include <assert.h>
#include <time.h>

#define TIME
//#define COMPDEBUG
//#define DEBUG

//#define DEFAULT_FILE_OUTPUT
#define PRINT_STATUS
//#define ANALYSIS
#define SHORT_ANALYSIS

#define EPSILON 0.005
#define MAIN_GRID_DIM 256

#define FAST

const float Pi = 3.1415926535;

typedef struct ComponentList ComponentList;
typedef struct ComponentNode ComponentNode;

struct ComponentList
{
	int componentSize;
	
	ComponentNode *head;
};

struct ComponentNode
{
	ComponentNode *next;
	
	int vertex;
};


ComponentList* initComponentList()
{
	ComponentList* list = (ComponentList*)malloc(sizeof(ComponentList));
	
	#ifdef COMPDEBUG
	printf("initComponentList - list: %p\n", list);
	fflush(stdout);
	#endif
	
	list->head = 0;
	list->componentSize = 0;
	
	return list;
}

void addComponentNode(ComponentList *list, int id)
{
	#ifdef COMPDEBUG
	printf("\nEntering addComponentNode - list %p - id: %d\n", list, id);
	fflush(stdout);
	#endif
	
	// Check if the list is empty
	if(list->head == 0)
	{
		// It is, then create a new node and link it to it
		#ifdef COMPDEBUG
		printf("list empty, filling it\n");
		fflush(stdout);
		#endif
		
		list->head = (ComponentNode*)malloc(sizeof(ComponentNode));
		(list->head)->next = 0;
		(list->head)->vertex = id;
		list->componentSize += 1;
	} else
	{
		#ifdef COMPDEBUG
		printf("list not empty, creating new head\n");
		fflush(stdout);
		#endif
		// It is not. Add to the head.
		ComponentNode* tempNode = (ComponentNode*)malloc(sizeof(ComponentNode));
		tempNode->vertex = id;
		tempNode->next = list->head;
		list->head = tempNode;
		list->componentSize += 1;
	}
	#ifdef COMPDEBUG
	printf("Exiting addComponentNode - list->componentSize: %d\n", list->componentSize);
	#endif
}

void destroyComponentList(ComponentList *list)
{
	if(list == 0)
		return;
		
	ComponentNode * tempNodeDestroy = list->head;
	ComponentNode *	tempNodeNext = list->head;
	
	while(tempNodeNext != 0)
	{
		tempNodeNext = tempNodeNext->next;
		free(tempNodeDestroy);
		tempNodeDestroy = tempNodeNext;
	}
	
	free(list);
}

		

__global__ void KuraFunc(float *da, float *dOutput, float *dInputTheta, float *dOmega, float K, int N);
__global__ void addMul(float *dOutput, float *dInput1, float *dInput2, float scalar1, float scalar2, int N);
__global__ void RK4(float *dTheta, float *dK1, float *dK2, float *dK3, float *dK4, float stepSize, int N);
__global__ void recRK4(float *dTheta, float *dK1, float *dK2, float *dK3, float *dK4, float stepSize, float *dThetaHist, float *dThetaDotHist, int i, int N);

__global__ void DistMatrixKernel(float * dD, float * dA, float * dThetaList, int N, int size);
__global__ void OttMatrixKernel(float * dDReal, float * dDImag, float * dA, float * dThetaHist, int N, int size);
__global__ void CalcRhoKernel(float * dRho, float * dThetaList, int N, int size);

void getDeviceInfo();

float randoCauchy(float mean, float spread);
float randoFlat(float minVal, float maxVal);
float randoGauss(float mean, float spread);

int randoInt(int minVal, int maxVal);

float * genErdos(int N, float probability);
float * genConfig(int N, float avgDegree, float power);
float * genAllToAll(int N);
float * genGardenes(int N, float alpha, int m0, int avgDegree);
float * genBarabasi(int initial_node_count, int final_node_count, float probability);

int BAConnect(int *degDist, int N, double m, double alpha);
void addLink(float *a, int i, int j, int N, int *degDist);

float * dCalcDistMatrix(float * dA, float * dThetaList, int N, int size);
float * hCalcDistMatrix(float *hA, float *hThetaList, int N, int size);

igraph_t *getSynchronizedGraph(float *hDistMatrix, float rLink, int N);

float * dCalcRho(float * dThetaList, int N, int size);
float * hCalcRho(float * hThetaHist, int N, int size);

float calcROtt(float *hA, float *dA, float *dThetaHist, int N, int size);
float calcRLink(float * a, float * D, int N);
float calcOrderParamMag(float * hThetaList, int N, int size);

float * calcDj(float *hA, float *dA, float *dThetaHist, int N, int size);

void checkCUDAError(int line);

void averageVector(float *avgVector, float *vector, int N, int size);

int * degreeDistribution(float * a, int N);

double largest_eigenvalue(float * a, int N);

void wrongSyntax();



int main(int argc, char **argv)
{
	
	int N;
	int size;
	int seed;
	int hist = 0;
	int m0;
	
	float K;
	
	double stepSize;
	double endTime;
	
	float erdosProb;
	
	float avgDegree;
	float power;
	float alpha;
	
	double lambda;
	FILE *oFile;
	
	float *a;
	
	igraph_t *syncGraph;
	igraph_vector_t sizeOfComponents;
	igraph_integer_t numOfComponents;
	igraph_vector_t membershipVector;
	
	int giantComponentSize;
	int numSyncComponents;
	int numSingletons;
	
	float (*ptrOmegaDistribution)(float, float) = NULL;
	float firstParam, secondParam;
	
	clock_t startProgram = clock();
	clock_t endProgram;
	
	
	
		#ifdef TIME
			cudaEvent_t start, stop;
			float et;
		#endif
	
	if(argc == 2)
	{
		if(strcmp("--info", argv[1]) != 0)
		{
			wrongSyntax();
			return 1;
		} else
		{
			getDeviceInfo();
			return 0;
		}
	} else if(argc == 11 || argc == 12 || argc == 13 || argc == 14)
	{
		// Grab the arguments from argv
		sscanf(argv[1], "%d", &N);
		sscanf(argv[2], "%f", &K);
		sscanf(argv[3], "%lf", &stepSize);
		sscanf(argv[4], "%lf", &endTime);
		sscanf(argv[5], "%d", &seed);
		sscanf(argv[7], "%f", &firstParam);
		sscanf(argv[8], "%f", &secondParam);
		
		printf("Number of oscillators: %d\nCoupling Constant: %g\nStep Size: %g\nEnd Time: %f\nFrequency Distribution: %s\nMean/Min: %f\nSpread/Max: %f\nNetwork Creation Method: %s\n", N, K, stepSize, endTime, argv[6], firstParam, secondParam, argv[9]);

		// Seed the Random Number Generator
		srand(seed);
		size = (int)(0.5*endTime/stepSize);
		
		if(strcmp("cauchy", argv[6]) == 0)
		{
			ptrOmegaDistribution = &randoCauchy;
		} else if(strcmp("gauss", argv[6]) == 0)
		{
			ptrOmegaDistribution = &randoGauss;
		} else if(strcmp("flat", argv[6]) == 0)
		{
			ptrOmegaDistribution = &randoFlat;
		} else
		{
			wrongSyntax();
			return 1;
		}
		
		// Need to check if --erdos_renyi or whatever we're doing is in the correct format
		if(strcmp("--erdos_renyi", argv[9]) == 0)
		{
			sscanf(argv[10], "%f", &erdosProb);
			oFile = fopen(argv[11], "w");
			a = genErdos(N, erdosProb);
			printf("Probability of Connection: %f\nOutput File: %s\n", erdosProb, argv[11]);
			
		} else if(strcmp("--all_to_all", argv[9]) == 0) // Create an all to all
		{
			oFile = fopen(argv[10], "w");
			a = genAllToAll(N);
			printf("Output File: %s\n", argv[10]);
			
		} else if(strcmp("--config_model", argv[9]) == 0)
		{
			sscanf(argv[10], "%f", &avgDegree);
			sscanf(argv[11], "%f", &power);
			oFile = fopen(argv[12], "w");
			a = genConfig(N, avgDegree, power);
			printf("Average Degree: %f\nPower Law: %f\nOutput File: %s\n", avgDegree, power, argv[12]);
			
		} else if(strcmp("--gardenes", argv[9]) == 0)
		{	
			sscanf(argv[10], "%f", &alpha);
			sscanf(argv[11], "%d", &m0);
			sscanf(argv[12], "%f", &avgDegree);
			oFile = fopen(argv[13], "w");
			a = genGardenes(N, alpha, m0, (int)(avgDegree/2));
			printf("Alpha: %f\nm0: %d\nAverage Degree: %f\nOutput File: %s\n", alpha, m0, avgDegree, argv[13]);
		} else if(strcmp("--barabasi", argv[9]) == 0)
		{
			sscanf(argv[10], "%d", &m0);
			sscanf(argv[11], "%f", &erdosProb);
			oFile = fopen(argv[12], "w");
			a = genBarabasi(m0, N, erdosProb);
			printf("Initial Node Count: %d\nErdos-Renyi Probability: %f\nOutput File: %s\n", m0, erdosProb, argv[12]);
		} else
		{
			printf("\nNot a valid network creation method.\n");
			return 1;
		}

	} else
	{
		wrongSyntax();
		return 1;
	}
		
	
	cudaDeviceProp deviceProp;
	
	cudaGetDeviceProperties(&deviceProp, 0);
	
	
	// First create host arrays, then initialize them
	printf("Calculated memory usage on Host: %u bytes\n", (unsigned int)(5*N*sizeof(float)+3*N*N*sizeof(float)+2*N*size*sizeof(float)));
	printf("Calculated memory usage on Device: %u bytes\n", (unsigned int)(9*N*sizeof(float) + 3*N*N*sizeof(float) + 2*N*size*sizeof(float)));
	printf("Device used: %s\n", deviceProp.name);
	printf("Time steps averaged over: %d\n", size);

			#ifdef PRINT_STATUS
				printf("Creating host arrays...");
			#endif	
	
	float *hTheta = (float*)malloc(N*sizeof(float));
	float *hOmega = (float*)malloc(N*sizeof(float));
	float *hTemp = (float*)malloc(N*sizeof(float));
	float *hDistMatrix = (float*)malloc(N*N*sizeof(float));
	float *hThetaHist = (float*)malloc(N*size*sizeof(float));
	float *hRho = (float*)malloc(N*N*sizeof(float));
	float *hThetaDotHist = (float*)malloc(N*size*sizeof(float));
	float *hAvgThetaDot = (float*)malloc(N*sizeof(float));
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	

			#ifdef DEBUG
				putchar('\n');
			#endif
	
			#ifdef DEFAULT_FILE_OUTPUT
				float *hostDistMatrix;
				float *hostRho;
				fprintf(oFile, "Theta Values: \n");
			#endif
	
			#ifdef PRINT_STATUS
				printf("Filling Theta and Omega...");
			#endif	
	
	for(int i = 0; i < N; i++)
	{
		hTheta[i] = randoFlat(0, 2*Pi);
		hOmega[i] = (*ptrOmegaDistribution)(firstParam, secondParam);
		
			#ifdef DEFAULT_FILE_OUTPUT
				fprintf(oFile, "%f\n", hTheta[i]);
			#endif
	}
	
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	
			#ifdef DEFAULT_FILE_OUTPUT
				fprintf(oFile, "\nOmega Values: \n");
				for(int i = 0; i < N; i++)
				{
					fprintf(oFile, "%f\n", hOmega[i]);
				}
	
				fprintf(oFile, "\nAdjacency Matrix: \n");
		
				for(int i = 0; i < N; i++)
				{
					for(int j = 0; j < N; j++)
					{
						fprintf(oFile, "%d ", (int)a[i*N + j]);
					}
			
					fprintf(oFile, "\n");
				}
			#endif 
	
	// Alright, now we should probably put these arrays on the device
	// Create device pointers
	float *da = 0;
	float *dTheta = 0;
	float *dOmega = 0;
	float *dTemp = 0;
	float *dK1 = 0;
	float *dK2 = 0;
	float *dK3 = 0;
	float *dK4 = 0;
	float *dThetaHist = 0;
	float *dDistMatrix = 0;
	float *dRho = 0;
	float *dThetaDotHist = 0;
	
			#ifdef PRINT_STATUS
				printf("Allocating space on device...");
			#endif	
	
	// Allocate device memory
	cudaMalloc((void**)&dTheta, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dOmega, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&da, N*N*sizeof(float));
	checkCUDAError(__LINE__);
	
	// Copy arrays to device memory
	cudaMemcpy(dTheta, hTheta, N*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError(__LINE__);
	cudaMemcpy(dOmega, hOmega, N*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError(__LINE__);
	cudaMemcpy(da, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError(__LINE__);
	
	// Allocate a temporary array to store...temporary values, and also the k1 values for each timestep
	cudaMalloc((void**)&dTemp, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dK1, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dK2, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dK3, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dK4, N*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dThetaHist, N*size*sizeof(float));
	checkCUDAError(__LINE__);
	cudaMalloc((void**)&dThetaDotHist, N*size*sizeof(float));
	checkCUDAError(__LINE__);
	
			#ifdef PRINT_STATUS
				printf("DONE\nRunning Main Loop...\n");
			#endif	
	
	// Now I guess we should start the main loop
	
	dim3 dimBlock(MAIN_GRID_DIM, 1, 1);
	dim3 dimGrid(((N % MAIN_GRID_DIM) == 0)?(N/MAIN_GRID_DIM):((int)(N/MAIN_GRID_DIM) + 1), 1, 1);
	
			#ifdef DEBUG
				printf("Main Grid Dimensions: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
				printf("Main Block Dimensions: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
			#endif
			
			#ifdef TIME
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);
			#endif
			
	
	for(double t = 0.0; t <= endTime; t += stepSize)
	{
		KuraFunc<<<dimGrid, dimBlock>>>(da, dK1, dTheta, dOmega, K, N);
		addMul<<<dimGrid, dimBlock>>>(dTemp, dTheta, dK1, 1.0, 0.5*stepSize, N);
		KuraFunc<<<dimGrid, dimBlock>>>(da, dK2, dTemp, dOmega, K, N);
		addMul<<<dimGrid, dimBlock>>>(dTemp, dTheta, dK2, 1.0, 0.5*stepSize, N);
		KuraFunc<<<dimGrid, dimBlock>>>(da, dK3, dTemp, dOmega, K, N);
		addMul<<<dimGrid, dimBlock>>>(dTemp, dTheta, dK3, 1.0, stepSize, N);
		KuraFunc<<<dimGrid, dimBlock>>>(da, dK4, dTemp, dOmega, K, N);
		
		if((endTime - t)/stepSize < (double)size)
		{
			recRK4<<<dimGrid, dimBlock>>>(dTheta, dK1, dK2, dK3, dK4, stepSize, dThetaHist, dThetaDotHist, hist, N);
			hist++;
		} else
		{
			RK4<<<dimGrid, dimBlock>>>(dTheta, dK1, dK2, dK3, dK4, stepSize, N);
		}
		
		if(cudaGetLastError() != cudaSuccess)
		{	
			printf("CUDA Error - Line %d - Time %f\n", __LINE__, t);
			break;
		}
	}
	
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	
			#ifdef TIME
				cudaEventRecord(stop,0);
				cudaEventSynchronize(stop);

				cudaEventElapsedTime(&et, start, stop);
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
	
				printf("Elapsed time of main loop: %g ms\n", et);
			#endif
	
			#ifdef PRINT_STATUS
				printf("Creating Rho and DistMatrix...");
			#endif	

	
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
				printf("Copying dTheta, dThetaHist, dDistMatrix, dRho, and dThetaDotHist to host...");
			#endif	
	
	cudaMemcpy(hTheta, dTheta, N*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError(__LINE__);
	cudaMemcpy(hThetaHist, dThetaHist, N*size*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError(__LINE__);
	
	// Free everything that we don't need anymore. Save memory!
	cudaFree(dTheta);
	cudaFree(dOmega);
	cudaFree(dTemp);
	cudaFree(dK1);
	cudaFree(dK2);
	cudaFree(dK3);
	cudaFree(dK4);
	
	
	// Grab Theta, Dist Matrix, and Rho, and hThetaList
	dRho = dCalcRho(dThetaHist, N, size);
	cudaMemcpy(hRho, dRho, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError(__LINE__);
	cudaFree(dRho);
	
	dDistMatrix = dCalcDistMatrix(da, dThetaHist, N, size);
	
	cudaMemcpy(hDistMatrix, dDistMatrix, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError(__LINE__);
	cudaFree(dDistMatrix);
	
	cudaMemcpy(hThetaDotHist, dThetaDotHist, N*size*sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError(__LINE__);
	
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif
	
			#ifdef DEFAULT_FILE_OUTPUT
				hostDistMatrix = hCalcDistMatrix(a, hThetaHist, N, size);
				hostRho = hCalcRho(hThetaHist, N, size);
			#endif
	
			#ifdef PRINT_STATUS
				printf("Averaging hThetaDotHist...");
			#endif	
	
	averageVector(hAvgThetaDot, hThetaDotHist, N, size);
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	// Now print out everything
	
			#ifdef PRINT_STATUS
				printf("Printing detailied information to files...");
			#endif	
			#ifdef DEFAULT_FILE_OUTPUT
				fprintf(oFile, "\nEnd Theta:\n");
		
				for(int i = 0; i < N; i++)
					fprintf(oFile, "%f ", hTheta[i]);
			
				fprintf(oFile, "\nTheta History:\n");
		
				for(int j = 0; j < size; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%f ", hThetaHist[j*N + i]);
					}
			
					fprintf(oFile, "\n");
				}
		
				fprintf(oFile, "\nTheta Dot History:\n");
		
				for(int j = 0; j < size; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%g ", hThetaDotHist[j*N + i]);
					}
					putc('\n', oFile);
				}
		
				fprintf(oFile, "\nDistance Matrix:\n");
		
				for(int j = 0; j < N; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%f ", hDistMatrix[j*N + i]);
					}
					fprintf(oFile, "\n");
				}
		
				fprintf(oFile, "\nHost Distance Matrix: \n");
		
				for(int j = 0; j < N; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%f%s ", hostDistMatrix[j*N + i], (((((hostDistMatrix[j*N + i] - hDistMatrix[j*N + i])/hostDistMatrix[j*N + i]) < EPSILON) || (hostDistMatrix[j*N + i] == hDistMatrix[j*N + i]))?"o":"!"));
					}
					fprintf(oFile, "\n");
				}
		
		
				fprintf(oFile, "\nDevice Rho: \n");
		
				for(int j = 0; j < N; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%f ", hRho[j*N + i]);
					}
					putc('\n', oFile);
				}
		
				fprintf(oFile, "\nHost Rho: \n");
		
				for(int j = 0; j < N; j++)
				{
					for(int i = 0; i < N; i++)
					{
						fprintf(oFile, "%f%s ", hostRho[j*N + i], (((((hostRho[j*N + i] - hRho[j*N + i])/hostRho[j*N + i]) < EPSILON) || (hostRho[j*N + i] == hRho[j*N + i]))?"o":"!"));
					}
					putc('\n', oFile);
				}
		
				fprintf(oFile, "\nOmega: \n");
		
				for(int j = 0; j < N; j++)
				{
					fprintf(oFile, "%f ", hAvgThetaDot[j]);
				}
			#endif
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	
			#ifdef PRINT_STATUS
				printf("Calculating ROtt...");
			#endif	
			
	float rOtt = calcROtt(a, da, dThetaHist, N, size);
	
			#ifdef PRINT_STATUS
				printf("DONE\nCalculating OrderParam...");
			#endif	
			
	float orderParam = calcOrderParamMag(hThetaHist, N, size);
	
			#ifdef PRINT_STATUS
				printf("DONE\nCalculating RLink...");
			#endif	
			
	float rLink = calcRLink(a, hDistMatrix, N);
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif
			
			#ifdef PRINT_STATUS
				printf("Calculating Dj...");
			#endif
			
	float *hDj = calcDj(a, da, dThetaHist, N, size);
	
			#ifdef PRINT_STATUS
				printf("DONE\nCalculating Largest Eigenvalue...");
			#endif
			
	lambda = largest_eigenvalue(a, N);
	
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif
	
			#ifdef PRINT_STATUS
				printf("Calculating Component Number and Giant Component Size...");
			#endif
			
	syncGraph = getSynchronizedGraph(hDistMatrix, rLink, N);
	igraph_vector_init(&sizeOfComponents, N);
	igraph_vector_init(&membershipVector, N);
	igraph_clusters(syncGraph, &membershipVector, &sizeOfComponents, &numOfComponents, IGRAPH_STRONG);
	giantComponentSize = (int)igraph_vector_max(&sizeOfComponents);
	
	numSyncComponents = 0;
	numSingletons = 0;
	
	for(int i = 0; i < (int)igraph_vector_size(&sizeOfComponents); i++)
	{
		if(VECTOR(sizeOfComponents)[i] > 1)
			numSyncComponents++;
		else
			numSingletons++;
	}
	
	#ifdef COMPDEBUG
	
	printf("\nigraph_vector_size(&membershipVector) - Should be N: %d\nnumOfComponents: %d\n(int)igraph_vector_size(&sizeOfComponents): %d\n", (int)igraph_vector_size(&membershipVector), (int)numOfComponents, (int)igraph_vector_size(&sizeOfComponents));
	
	for(int i = 0; i < (int)igraph_vector_size(&membershipVector); i++)
	{
		printf("%d ", (int)VECTOR(membershipVector)[i]);
		
		if((i + 1) % 10 == 0)
			putchar('\n');
	}
	
	putchar('\n');
	
	fflush(stdout);
	#endif
	
	ComponentList ** compList = (ComponentList**)malloc(numOfComponents*sizeof(ComponentList*));
	
	#ifdef COMPDEBUG
	printf("Initializing compList...");
	fflush(stdout);
	#endif
	
	for(int i = 0; i < (int)igraph_vector_size(&sizeOfComponents); i++)
	{
		compList[i] = initComponentList();
	}
	
	#ifdef COMPDEBUG
	for(int i = 0; i < (int)igraph_vector_size(&sizeOfComponents); i++)
	{
		printf("\n i - %d : compList[i] - %p", i, compList[i]);
	}
	
	printf("DONE\n");
	fflush(stdout);
	
	printf("Adding components...");
	fflush(stdout);
	#endif
	
	for(int i = 0; i < (int)igraph_vector_size(&membershipVector); i++)
	{
		// Need to create a list of components and add each vertex to their list of respective components
		#ifdef COMPDEBUG
		printf("i: %d (int)VECTOR(membershipVector)[i]: %d VECTOR(membershipVector)[i]: %g\n", i, (int)VECTOR(membershipVector)[i], VECTOR(membershipVector)[i]);
		fflush(stdout);
		#endif
		addComponentNode(compList[((int)VECTOR(membershipVector)[i])], i);
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	float * averageFrequency = (float*)malloc(numOfComponents*sizeof(float));
	ComponentNode * tempNode;
	
	#ifdef COMPDEBUG
	printf("Finding average frequency...");
	fflush(stdout);
	#endif
	
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		averageFrequency[i] = 0;
		
		tempNode = (compList[i])->head;
		
		while(tempNode != 0)
		{
			averageFrequency[i] += hOmega[tempNode->vertex];
			tempNode = tempNode->next;
		}
		
		averageFrequency[i] /= (float)((compList[i])->componentSize);
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	float * varFreq = (float*)malloc(numOfComponents*sizeof(float));
	
	#ifdef COMPDEBUG
	printf("Finding varFreq...");
	fflush(stdout);
	#endif
	
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		varFreq[i] = 0;
		
		// Now sum over the square of the frequency differences
		tempNode = (compList[i])->head;
		
		while(tempNode != 0)
		{
			varFreq[i] += (hOmega[tempNode->vertex] - averageFrequency[i])*(hOmega[tempNode->vertex] - averageFrequency[i]);
			tempNode = tempNode->next;
		}
		
		varFreq[i] /= compList[i]->componentSize;
		
		varFreq[i] = sqrt(varFreq[i]);
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	// Time to find the number of specific type!
	// First start out by creating a list of the fours
	
	#ifdef COMPDEBUG
	printf("Finding number of threes...");
	fflush(stdout);
	#endif
	int numberOfThrees = 0;
	int temp = 0;
	
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		if(compList[i]->componentSize == 3)
			numberOfThrees += 1;
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	ComponentList ** threeList = (ComponentList**)malloc(numberOfThrees*sizeof(ComponentList*));
	
	#ifdef COMPDEBUG
	printf("Creating three list...");
	fflush(stdout);
	#endif
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		if(compList[i]->componentSize == 3)
		{
			threeList[temp] = compList[i];
			temp += 1;
		}
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	int threeType[2] = {0};
	
	int numLinks;
	ComponentNode *searchNode;
	
	#ifdef COMPDEBUG
	printf("Finding threeType...numberOfThrees - %d...", numberOfThrees);
	fflush(stdout);
	#endif
	for(int i = 0; i < numberOfThrees; i++)
	{
		tempNode = (threeList[i])->head;
		numLinks = 0;
		
		#ifdef COMPDEBUG
		printf("Iteration %d\n", i);
		fflush(stdout);
		#endif
		
		while(tempNode != 0)
		{
			searchNode = (threeList[i])->head;
			
			#ifdef COMPDEBUG
			printf("tempNode->vertex: %d\n", tempNode->vertex);
			fflush(stdout);
			#endif
			
			while(searchNode != 0)
			{
				#ifdef COMPDEBUG
				printf("searchNode->vertex: %d\n", searchNode->vertex);
				fflush(stdout);
				#endif
				
				if(searchNode != tempNode)
				{
					#ifdef COMPDEBUG
					printf("a - %d\n", (int)a[(searchNode->vertex)*N + (tempNode->vertex)]);
					fflush(stdout);
					#endif
					numLinks += (int)a[(searchNode->vertex)*N + (tempNode->vertex)];
				}
				
				searchNode = searchNode->next;
			}
			
			tempNode = tempNode->next;
		}
		
		if(numLinks % 2 != 0 || numLinks > 6)
			printf("\nSOMETHING HAS GONE TERRIBLY WRONG\n");
		
		#ifdef COMPDEBUG
		printf("numLinks - %d...\n", numLinks);
		fflush(stdout);
		#endif
		
		numLinks /= 2;
		
		threeType[numLinks - 2] += 1;
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	int numberOfFours = 0;
	
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		if(compList[i]->componentSize == 4)
			numberOfFours += 1;
	}
	
	ComponentList ** fourList = (ComponentList**)malloc(numberOfFours*sizeof(ComponentList*));
	
	temp = 0;
	
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		if(compList[i]->componentSize == 4)
		{
			fourList[temp] = compList[i];
			temp += 1;
		}
	}
		
	// Now we have our list of four components, now we must identify them
	
	int fourType[6] = {0};
	int highestDegree;
	int tempDegree;
	
	for(int i = 0; i < numberOfFours; i++)
	{
		// For each component, discover how many links it has and it's highest degree
		tempNode = (fourList[i])->head;
		numLinks = 0;
		highestDegree = 0;
		
		while(tempNode != 0)
		{
			searchNode = (fourList[i])->head;
			tempDegree = 0;
			
			#ifdef COMPDEBUG
			printf("tempNode->vertex: %d\n", tempNode->vertex);
			#endif
			
			while(searchNode != 0)
			{
				#ifdef COMPDEBUG
				printf("searchNode->vertex: %d\n", searchNode->vertex);
				#endif
				
				if(searchNode != tempNode)
				{
					#ifdef COMPDEBUG
					printf("a - %d\n", (int)a[(searchNode->vertex)*N + (tempNode->vertex)]);
					#endif
					
					numLinks += (int)a[(searchNode->vertex)*N + (tempNode->vertex)];
					tempDegree += (int)a[(searchNode->vertex)*N + (tempNode->vertex)];
				} 
					
				searchNode = searchNode->next;	
			}
			
			if(tempDegree > highestDegree)
				highestDegree = tempDegree;
			
			#ifdef COMPDEBUG
			printf("highestDegree - %d - tempDegree - %d\n", highestDegree, tempDegree);
			#endif
			
			tempNode = tempNode->next;	
		}
	
		if(numLinks % 2 != 0)
			printf("\nSOMETHING HAS GONE TERRIBLY WRONG\n");
		
		numLinks /= 2;
		
		switch(numLinks)
		{
			case 3:
				if(highestDegree == 2)
					fourType[0] += 1;
				else if(highestDegree == 3)
					fourType[1] += 1;
				else
					printf("\nSOMETHING HAS GONE TERRIBLY WRONG\n");
				break;
			case 4:
				if(highestDegree == 2)
					fourType[3] += 1;
				else if(highestDegree == 3)
					fourType[2] += 1;
				else
					printf("\nSOMETHING HAS GONE TERRIBLY WRONG\n");
				break;
			case 5:
				fourType[4] += 1;
				break;
			case 6:
				fourType[5] += 1;
				break;
			default:
				printf("\nSOMETHING HAS GONE TERRIBLY WRONG\n");
		}
	}
	
	#ifdef COMPDEBUG
	printf("Creating adjGraph...");
	fflush(stdout);
	#endif
	
	igraph_t *adjGraph = (igraph_t*)malloc(sizeof(igraph_t));
	igraph_real_t adjClustering;
	igraph_real_t syncClustering;
	
	igraph_vector_t adjEdges;
	
	igraph_empty(adjGraph, N, 0);
	
	igraph_vector_init(&adjEdges, 0);
	
	
	for(int i = 0; i < N; i++)
	{
		for(int j = i; j < N; j++)
		{
			if(a[i*N + j] == 1)
			{
				igraph_vector_push_back(&adjEdges, i);
				igraph_vector_push_back(&adjEdges, j);
			}
		}
	}
	
	igraph_add_edges(adjGraph, &adjEdges, 0);
	igraph_vector_destroy(&adjEdges);
	
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif
	
	igraph_transitivity_undirected(adjGraph, &adjClustering);
	igraph_transitivity_undirected(syncGraph, &syncClustering);
	
	
	
	free(averageFrequency);
	
	#ifdef COMPDEBUG
	printf("Destroying componentLists...");
	fflush(stdout);
	#endif
	for(int i = 0; i < (int)numOfComponents; i++)
	{
		destroyComponentList(compList[i]);
	}
	#ifdef COMPDEBUG
	printf("DONE\n");
	fflush(stdout);
	#endif

	free(compList);
	free(fourList);
	free(threeList);
	
	igraph_vector_destroy(&membershipVector);
	igraph_destroy(syncGraph);
	igraph_destroy(adjGraph);
	free(syncGraph);
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif
			
	printf("Order Parameter: %f\nRLink: %f\nROtt: %f\nLargest Eigenvalue: %f\n", orderParam, rLink, rOtt, lambda);
	printf("Giant Component Size: %d\nNumber of Components: %d\nNumber of Singletons: %d\n", giantComponentSize, numSyncComponents, numSingletons);
	
			#ifdef PRINT_STATUS
				printf("Printing analysis...");
			#endif	
			#ifdef ANALYSIS
				int *degDist = degreeDistribution(a, N);
				float omegs;
				float dots;
				
				for(int i = 0; i < N; i++)
				{
					for(int j = 0; j < N; j++)
					{
						if(i != j)
						{
							omegs = abs(hOmega[i] - hOmega[j]);
							dots =  hAvgThetaDot[i] - hAvgThetaDot[j];
							fprintf(oFile, "%f %f %f %f %f %f\n", hDistMatrix[i*N + j], hRho[i*N + j], omegs, dots, dots/omegs, omegs*dots);
						}
					}
				}
		
				putc('\n', oFile);
				putc('\n', oFile);
		
				for(int i = 0; i < N; i++)
				{
					fprintf(oFile, "%f %f\n", hOmega[i], hAvgThetaDot[i]);
				}
		
				fprintf(oFile, "\n%f %f %f %f\n", K, orderParam, rLink, rOtt);
				
				free(degDist);
			#endif
			
			#ifdef SHORT_ANALYSIS
				int *degDist = degreeDistribution(a, N);
				
				
				for(int i = 0; i < N; i++)
				{
					fprintf(oFile, "%f %f %f %d\n", hDj[i], hOmega[i], hAvgThetaDot[i], degDist[i]);
				}
				
				
					
				fprintf(oFile, "\n%f %f %f %f\n", K, orderParam, rLink, rOtt);
				fprintf(oFile, "%f %d %d %d\n", lambda, numSyncComponents, giantComponentSize, numSingletons);
				
				putc('\n', oFile);
				
				for(int i = 0; i < numOfComponents; i++)
				{
					if((int)VECTOR(sizeOfComponents)[i] != 1)
						fprintf(oFile, "%d %f 0 0\n", (int)VECTOR(sizeOfComponents)[i], varFreq[i]);
				}
				
				
				fprintf(oFile, "\n%f %f %d %d\n", (double)syncClustering, (double)adjClustering, threeType[0], threeType[1]);
				
				for(int i = 0; i < 6; i++)
				{
					fprintf(oFile, "%d ", fourType[i]);
				}
				putc('\n', oFile);
				
				free(degDist);
			#endif
			
			#ifdef PRINT_STATUS
				printf("DONE\n");
			#endif	
	
	endProgram = clock();
	
	printf("Total Run Time: %g s\n", (double)(endProgram - startProgram)/(double)CLOCKS_PER_SEC);
	checkCUDAError(__LINE__);
	
	igraph_vector_destroy(&sizeOfComponents);
	
	// Cleanup
	free(hTheta);
	free(hOmega);
	free(hTemp);
	free(a);
	free(hThetaHist);
	free(hDistMatrix);
	free(hRho);
	free(hThetaDotHist);
	free(hAvgThetaDot);
	free(hDj);
	free(varFreq);
	//free(hostRho);
	//free(hostDistMatrix);
	
	cudaFree(da);
	
	cudaFree(dThetaHist);
	cudaFree(dThetaDotHist);
	
	fclose(oFile);
	
	return 0;
}

__global__ void KuraFunc(float *da, float *dOutput, float *dInputTheta, float *dOmega, float K, int N)
{
	//Let the thread block be one dimensional, seperated by blocks so that index = blockIdx.x*512 + threadIdx.x
	int tIndex = blockIdx.x*blockDim.x + threadIdx.x;		// Each block can have up to 512 threads, so more won't be necessary, and we'll only dispatch as many threads as we have oscillators
	float theta;
	float result;
	
	if(tIndex < N)
	{
		theta = dInputTheta[tIndex];
		result = 0.0;
		
		for(int i = 0; i < N; i++)
		{
			#ifndef FAST
				result += da[tIndex*N+i]*sin(dInputTheta[i] - theta);
			#endif
		
			#ifdef FAST
				result += da[tIndex*N+i]*__sinf(dInputTheta[i] - theta);
			#endif
		}
	
		dOutput[tIndex] = dOmega[tIndex] + K*result;
	}
}

__global__ void addMul(float *dOutput, float *dInput1, float *dInput2, float scalar1, float scalar2, int N)
{
	int tIndex = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(tIndex < N)
		dOutput[tIndex] = scalar1*dInput1[tIndex] + scalar2*dInput2[tIndex];
}

__global__ void RK4(float *dTheta, float *dK1, float *dK2, float *dK3, float *dK4, float stepSize, int N)
{	
	int tIndex = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(tIndex < N)
		dTheta[tIndex] = dTheta[tIndex] + (stepSize*dK1[tIndex] + 2*stepSize*dK2[tIndex] + 2*stepSize*dK3[tIndex] + stepSize*dK4[tIndex])/6;
}


__global__ void recRK4(float *dTheta, float *dK1, float *dK2, float *dK3, float *dK4, float stepSize, float *dThetaHist, float *dThetaDotHist, int i, int N)
{
	int tIndex = blockIdx.x*blockDim.x + threadIdx.x;
	float thetaDot;
	
	if(tIndex < N)
	{
		thetaDot = (dK1[tIndex] + 2.0*dK2[tIndex] + 2.0*dK3[tIndex] + dK4[tIndex])/6.0;
	
		dTheta[tIndex] = dTheta[tIndex] + stepSize*thetaDot;
	
		dThetaHist[i*N + tIndex] = dTheta[tIndex];
		dThetaDotHist[i*N + tIndex] = thetaDot;
	}
}

void getDeviceInfo()
{
	int deviceCount = 0;
	int nGpuArchCoresPerSM[] = { -1, 8, 32 };

	
	if(cudaGetDeviceCount(&deviceCount) != cudaSuccess)
	{
		printf("\nUnfortunately could not get CUDA information.\n");
		return;
	}
	
	if(deviceCount == 0)
	{
		printf("\nNo CUDA supporting device found.\n");
		return;
	}
	
	int driverVersion = 0, runtimeVersion = 0;
	
	for(int dev = 0; dev < deviceCount; ++dev)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		
		if(dev == 0)
		{
			if(deviceProp.major == 9999 && deviceProp.minor == 9999)
				printf("\nThere is no CUDA supporting device.\n");
			else if(deviceCount == 1)
				printf("\nOne CUDA supporting device found.\n");
			else
				printf("\nThere are %d devices supporting CUDA\n", deviceCount);
		}
		
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		
		cudaDriverGetVersion(&driverVersion);
		printf("   CUDA Driver Version:                           %d\n", driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("   CUDA Runtime Version:                          %d\n", runtimeVersion);
		printf("   CUDA Capability Major revision number:         %d\n", deviceProp.major);
		printf("   CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
		printf("   Total amount of global memory: 	          %u bytes\n", (unsigned int)deviceProp.totalGlobalMem);
		printf("   Number of multiprocessors:		          %d\n", deviceProp.multiProcessorCount);
		printf("   Number of cores:			          %d\n", nGpuArchCoresPerSM[deviceProp.major]*deviceProp.multiProcessorCount);
		printf("   Total amount of constant memory: 	          %u bytes\n", (unsigned int)deviceProp.totalConstMem);
		printf("   Total amount of shared memory per block:       %u bytes\n", (unsigned int)deviceProp.sharedMemPerBlock);
		printf("   Total number of registers available per block: %u bytes\n", (unsigned int)deviceProp.regsPerBlock);
		printf("   Warp size:                                     %d\n", deviceProp.warpSize);
		printf("   Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("   Maximum sizes of each dimension of a block:    %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("   Maximum sizes of each dimension of a grid:     %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("   Maximum memory pitch:                          %u bytes\n", (unsigned int)deviceProp.memPitch);
		printf("   Texture alignment:				  %u bytes\n", (unsigned int)deviceProp.textureAlignment);
		printf("   Clock rate:                                    %.2f GHz\n\n", deviceProp.clockRate*1e-6f);
	

	}
}
	
		
float randoCauchy(float mean, float spread)
{
	double P = ((double)(rand()))/((double)(RAND_MAX));
	
	return mean + spread*tan(Pi*P);
}
	
float randoFlat(float minVal, float maxVal)
{
	return rand()*(maxVal - minVal)/((float)RAND_MAX) + minVal;
}

float randoGauss(float mean, float spread)
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return mean + spread*(float)(X)	;
}

float * genErdos(int N, float probability)
{
	float *a = (float*)malloc(N*N*sizeof(float));

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < i; j++)
		{
			if(randoFlat(0, 1) < probability)
			{
				a[i*N + j] = 1;
				a[j*N + i] = 1;
			} else
			{
				a[i*N + j] = 0;
				a[j*N + i] = 0;
			}
		}
		a[i*N + i] = 0;
	}

	#ifdef DEBUG
	printf("\nCreated a: \n");
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			printf("%d ", (int)a[i*N + j]);
		}

		putchar('\n');
	}
	#endif
		

	return a;
}

float * genConfig(int N, float avgDegree, float power)
{
	float normConst = 0.0;
	
	// Create a degree distribution
	float * v = (float*)malloc(N*sizeof(float));
	
	// Create adjacency matrix
	float * a = (float*)malloc(N*N*sizeof(float));
	
	for(int i = 0; i < N; i++)
	{
		v[i] = pow((i + 1), -1/(power - 1));
	}
	
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			normConst += v[i]*v[j];
			
	normConst = (0.5*avgDegree*N)/(normConst);
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < i; j++)
		{
			a[i*N + j] = (((normConst*v[i]*v[j])>randoFlat(0, 1))?1:0);
			a[j*N + i] = a[i*N + j];
		}
	}
	
	free(v);
	
	return a;
}


__global__ void DistMatrixKernel(float * dD, float * dA, float * dThetaList, int N, int size)
{
	// First find out where we are in the block/grid
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	float tempReal;
	float tempImag;
	
	if((i < N) && (j < N))
	{
		tempReal = 0.0;
		tempImag = 0.0;
		
		for(int k = 0; k < size; k++)
		{
			tempReal += cos(dThetaList[N*k + i] - dThetaList[N*k + j]);
			tempImag += sin(dThetaList[N*k + i] - dThetaList[N*k + j]);
		}
	
		tempReal /= size;
		tempImag /= size;
	
		dD[N*i + j] = dA[N*i + j]*sqrt(tempReal*tempReal + tempImag*tempImag);
	}
	
}

float * dCalcDistMatrix(float * dA, float * dThetaList, int N, int size)
{	
	// First create our D matrix, we want to do this on device
	float * dD;
	
	cudaMalloc((void**)&dD, N*N*sizeof(float));
	
	// Now we want to fill this matrix up. We want the amplitude of the averaged complex exponential between theta[i] and theta[j]
	// Assume dThetaList is structured in this way: It is N*size*sizeof(float) bytes large, thus, for each i and j, we want to loop over each of the [i] and [j] entries in dThetaList, that will be done in a call to a global function
	
	// Set up our dimensions for dispatch
	// Let us split everything up into a 16x16 blocks and each thread will calculate its place
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(((N % 16) == 0)?(N/16):((int)(N/16) + 1), ((N % 16) == 0)?(N/16):((int)(N/16) + 1), 1);
	// Global function call
	#ifdef DEBUG
		printf("dCalcDistMatrix Grid Dimensions: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("dCalcDistMatrix Block Dimensions: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	#endif
		
	DistMatrixKernel<<<dimGrid, dimBlock>>>(dD, dA, dThetaList, N, size);
	
	return dD;
}

float * hCalcDistMatrix(float *hA, float *hThetaList, int N, int size)
{
	float *hD = (float*)malloc(N*N*sizeof(float));
	float tempReal;
	float tempImag;
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			tempReal = 0.0;
			tempImag = 0.0;
			
			for(int k = 0; k < size; k++)
			{
				tempReal += cos(hThetaList[k*N + i] - hThetaList[k*N + j]);
				tempImag += sin(hThetaList[k*N + i] - hThetaList[k*N + j]);
			}
			
			tempReal /= size;
			tempImag /= size;
			
			hD[i*N + j] = hA[i*N + j]*sqrt(tempReal*tempReal + tempImag*tempImag);
		}
	}
	
	return hD;
}


__global__ void OttMatrixKernel(float * dDReal, float * dDImag, float * dA, float * dThetaHist, int N, int size)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	
	float tempReal;
	float tempImag;
	
	if((i < N) && (j < N))
	{
		tempReal = 0.0;
		tempImag = 0.0;
		
		for(int k = 0; k < size; k++)
		{
			tempReal += cos(dThetaHist[N*k + i] - dThetaHist[N*k + j]);
			tempImag += sin(dThetaHist[N*k + i] - dThetaHist[N*k + j]);
		}
	
		tempReal /= size;
		tempImag /= size;
	
		dDReal[N*i + j] = dA[N*i + j]*tempReal;
		dDImag[N*i + j] = dA[N*i + j]*tempImag;
	}
}

float calcROtt(float *hA, float *dA, float *dThetaHist, int N, int size)
{
	// cudaMalloc two arrays, one for the cos(Theta_I - Theta_J) and one for sin(Theta_I - Theta_J), one is the real and the other is the imaginary part
	float *dRealD;
	float *dImagD;
	float *hRealD = (float*)malloc(N*N*sizeof(float));
	float *hImagD = (float*)malloc(N*N*sizeof(float));
	
	float sumA = 0.0;
	float tempReal = 0.0;
	float tempImag = 0.0;
	
	cudaMalloc((void**)&dRealD, N*N*sizeof(float));
	cudaMalloc((void**)&dImagD, N*N*sizeof(float));
	
	// Now call the kernels to fill dRealD and dImagD
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(((N % 16) == 0)?(N/16):((int)(N/16) + 1), ((N % 16) == 0)?(N/16):((int)(N/16) + 1), 1);
	
	#ifdef DEBUG
		printf("calcROtt Grid Dimensions: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("calcROtt Block Dimensions: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	#endif
	
	OttMatrixKernel<<<dimGrid, dimBlock>>>(dRealD, dImagD, dA, dThetaHist, N, size);
	
	cudaMemcpy(hRealD, dRealD, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hImagD, dImagD, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	// Now have both dRealD and dImagD, add them seperately
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			tempReal += hRealD[i*N + j];
			tempImag += hImagD[i*N + j];
			sumA += hA[i*N + j];
		}
	}
	
	cudaFree(dRealD);
	cudaFree(dImagD);
	
	free(hRealD);
	free(hImagD);
	
	return sqrt(tempReal*tempReal + tempImag*tempImag)/sumA;
}
			
			
float * genAllToAll(int N)
{
	float * a = (float*)malloc(N*N*sizeof(float));
	
	for(int i = 0; i < N*N; i++)
	{
		a[i] = 1;
	}
	
	return a;
}

float calcRLink(float * a, float * D, int N)
{
	float temp = 0.0;
	float twoN = 0.0;
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			temp += D[N*i+j];
			twoN += a[N*i+j];
		}
	}
	
	return temp/twoN;
}

__global__ void CalcRhoKernel(float * dRho, float * dThetaList, int N, int size)
{
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	float temp;
	
	if((i < N) && (j < N))
	{
		temp = 0.0;
		// Now calculate cos(theta_i - theta_j) and add them
		for(int k = 0; k < size; k++)
		{
			temp += cos(dThetaList[k*N + i] - dThetaList[k*N + j]);
		}
	
		temp /= size;
	
		dRho[N*i + j] = temp;
	}
}
	
float * dCalcRho(float * dThetaList, int N, int size)
{
	// Want to create a new matrix on the device
	float * dRho;
	
	cudaMalloc((void**)&dRho, N*N*sizeof(float));
	
	// Now dimensions!
	
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(((N % 16) == 0)?(N/16):((int)(N/16) + 1), ((N % 16) == 0)?(N/16):((int)(N/16) + 1), 1);
	
	#ifdef DEBUG
		printf("dCalcDistMatrix Grid Dimensions: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("dCalcDistMatrix Block Dimensions: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	#endif
	
	//printf("Rho dimBlock: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	//printf("Rho dimGrid: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
	// Now initiate kernel
	CalcRhoKernel<<<dimGrid, dimBlock>>>(dRho, dThetaList, N, size);
	
	return dRho;
}

float * hCalcRho(float * hThetaHist, int N, int size)
{
	float *hostRho = (float*)malloc(N*N*sizeof(float));
	float temp;
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			temp = 0.0;
			
			for(int k = 0; k < size; k++)
			{
				temp += cos(hThetaHist[k*N + i] - hThetaHist[k*N + j]);
			}
			
			temp /= size;
			hostRho[i*N + j] = temp;
		}
	}
	
	return hostRho;
}
	

float calcOrderParamMag(float * hThetaList, int N, int size)
{
	float tempReal = 0.0;
	float tempImag = 0.0;
	float tempMag = 0.0;
	
	for(int i = 0; i < size; i++)
	{
		tempReal = 0.0;
		tempImag = 0.0;
		
		for(int j = 0; j < N; j++)
		{
			tempReal += cos(hThetaList[i*N + j]);
			tempImag += sin(hThetaList[i*N + j]);
		}
		tempMag += (sqrt(tempReal*tempReal + tempImag*tempImag)/N);
	}
	
	return tempMag/size;
}

void checkCUDAError(int line)
{
	cudaError_t erra = cudaGetLastError();
	
	if(cudaSuccess != erra)
	{
		printf("CUDA Error - Line %d: %s\n", line, cudaGetErrorString(erra));
	}
}


void averageVector(float *avgVector, float *vector, int N, int size)
{
	double tempVal;
	
	
	for(int i = 0; i < N; i++)
	{
		tempVal = 0.0;
		for(int j = 0; j < size; j++)
		{	
			tempVal += (double)vector[j*N + i];
		}
		tempVal /= (double)size;
		
		avgVector[i] = (float)tempVal;
	}
	
}



int * degreeDistribution(float * a, int N)
{
	int *degDist = (int*)malloc(N*sizeof(int));
	
	for(int i = 0; i < N; i++)
	{
		degDist[i] = 0;
		
		for(int j = 0; j < N; j++)
		{
			degDist[i] += (int)a[i*N + j];
		}
	}
	
	return degDist;
}

float * calcDj(float *hA, float *dA, float *dThetaHist, int N, int size)
{
	float *dRealD;
	float *dImagD;
	float *hRealD = (float*)malloc(N*N*sizeof(float));
	float *hImagD = (float*)malloc(N*N*sizeof(float));
	
	float tempReal;
	float tempImag;
	float sumA;
	
	float *Dj = (float*)malloc(N*sizeof(float));
	
	cudaMalloc((void**)&dRealD, N*N*sizeof(float));
	cudaMalloc((void**)&dImagD, N*N*sizeof(float));
	
	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(((N % 16) == 0)?(N/16):((int)(N/16) + 1), ((N % 16) == 0)?(N/16):((int)(N/16) + 1), 1);
	
	#ifdef DEBUG
		printf("dCalcDistMatrix Grid Dimensions: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("dCalcDistMatrix Block Dimensions: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
	#endif
	
	OttMatrixKernel<<<dimGrid, dimBlock>>>(dRealD, dImagD, dA, dThetaHist, N, size);
	
	cudaMemcpy(hRealD, dRealD, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hImagD, dImagD, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < N; i++)
	{
		tempReal = 0.0;
		tempImag = 0.0;
		sumA = 0.0;
		
		for(int j = 0; j < N; j++)
		{
			tempReal += hRealD[i*N + j];
			tempImag += hImagD[i*N + j];
			sumA += hA[i*N + j];
		}
		
		Dj[i] = sqrt(tempReal*tempReal + tempImag*tempImag)/sumA;
	}
		
	
	cudaFree(dRealD);
	cudaFree(dImagD);
	
	free(hRealD);
	free(hImagD);
	
	return Dj;
	
}

double largest_eigenvalue(float * a, int N)
{
	double *v = (double*)malloc(N*sizeof(double));
	double *v_prime = (double*)malloc(N*sizeof(double));

	double mag_result = 0;
	double dot_result;
	
	for(int i = 0; i < N; i++)
	{	
		v[i] = randoFlat(-1, 1);
		mag_result += v[i]*v[i];
	}

	mag_result = sqrt(mag_result);

	for(int i = 0; i < N; i++)
		v[i] /= mag_result;
	

	do
	{
		mag_result = 0;
		dot_result = 0;

		for(int i = 0; i < N; i++)
		{
			v_prime[i] = 0;

			for(int j = 0; j < N; j++)
				v_prime[i] += ((double)a[j*N + i])*v[j];

			dot_result += v_prime[i]*v[i];
			mag_result += v_prime[i]*v_prime[i];
		}

		mag_result = sqrt(mag_result);
		dot_result /= mag_result;

		if(v_prime[0] < 0)
			mag_result = -mag_result;

		for(int i = 0; i < N; i++)
			v[i] = v_prime[i]/mag_result;
	} while((1.0 - dot_result) > 1.0e-12);

	return  mag_result;
}

int randoInt(int minVal, int maxVal)
{
	//return ((int)rand()) % (maxVal - minVal + 1) + minVal;
	return minVal + (int)(rand()*((maxVal - minVal + 1)/(RAND_MAX + 1.0)));
}

void addLink(float *a, int i, int j, int N, int *degDist)
{	
	if((a[i*N + j] == 0) && (a[j*N + i] == 0))
	{
		degDist[i] += 1;
		degDist[j] += 1;
	} 
	
	a[i*N + j] = 1.0;
	a[j*N + i] = 1.0;

}



// Given a degree distribution, and N, return a random index weighted by degree - those with zero degree have zero chance connecting
int BAConnect(int *degDist, int N, double m, double alpha)
{
	double *prob = (double*)malloc(N*sizeof(double));
	double sum = 0.0;
	double accumulate = 0.0;
	double randNum = rand()/((double)RAND_MAX);
	
	#ifdef GARDEBUG
		fprintf(oFile, "\nEntering BAConnect\nReceived Degree Distribution:\n");
		for(int i = 0; i < N; i++)
			fprintf(oFile, "%d %d\n", i, degDist[i]);
	#endif
	
	for(int i = 0; i < N; i++)
	{
		if(degDist[i] != 0)
			prob[i] = (double)(degDist[i]) + m;
		else
			prob[i] = 0;
		sum += prob[i];
	}
	
	
	for(int i = 0; i < N; i++)
	{
		prob[i] /= sum;
	}
	
	#ifdef GARDEBUG
		fprintf(oFile, "\nProbability Distribution:\n");
		for(int i = 0; i < N; i++)
			fprintf(oFile, "%d %g\n", i, prob[i]);
	#endif
	
	
	for(int i = 0; i < N; i++)
	{
		accumulate += prob[i];
		
		if(accumulate > randNum)
		{
			return i;
		}
	}
	
	return N - 1;
}

float * genGardenes(int N, float alpha, int m0, int avgDegree)
{
	#ifdef GARDEBUG
		printf("Entered genGardenes: %d %f %d %d\n", N, alpha, m0, avgDegree);
	#endif
	
	int *Omega = (int*)malloc(N*sizeof(int));
	int *Unconnected = (int*)malloc(N*sizeof(int));
	int *Connected = (int*)malloc(N*sizeof(int));
	
	#ifdef GARDEBUG
		printf("Allocated Omega, Unconnected and Connected.\n");
		oFile = fopen("DEBUG_GARDENES.txt", "w");
	#endif
	
	int sizeU = 0;
	
	int i, j;
	
	int tempInt, tempIndex;
	
	float *a = (float*)malloc(N*N*sizeof(float));
	int *degDist = (int*)malloc(N*sizeof(int));
	
	#ifdef GARDEBUG
		printf("Allocated a, degDist.\n");
	#endif
	
	for(int k = 0; k < N; k++)
	{
		degDist[k] = 0;
		// Create Omega
		
		Omega[k] = 1;
		Unconnected[k] = 0;
		Connected[k] = 0;
		
		for(int l = k; l < N; l++)
		{
			a[k*N + l] = 0;
			a[l*N + k] = 0;
		}
	}
	
	#ifdef GARDEBUG
		printf("Initialized everything.\n");
	#endif
	
	#ifdef GARDEBUG
		for(int k = 0; k < N; k++)
		{
			for(int l = 0; l < N; l++)
			{
				printf("%d ", (int)a[k*N + l]);
			}
			putchar('\n');
		}
	#endif
	
	
	for(int k = 0; k < m0; k++)
	{
		for(int l = 0; l < m0; l++)
		{
			if(k != l)
			{
				addLink(a, k, l, N, degDist);
			}
		}
		
		// Add node i to N
		Connected[k] = 1;
	}
	
	// We've now constructed N, now onto U
	
	for(int k = m0; k < N; k++)
	{
		Unconnected[k] = 1;
		sizeU++;
	}
	
	
	#ifdef GARDEBUG
		putchar('\n');
		for(int k = 0; k < N; k++)
		{
			for(int l = 0; l < N; l++)
			{
				printf("%d ", (int)a[k*N + l]);
			}
			putchar('\n');
		}
	#endif
	
	/*** 	The algorithm is as such:
		REPEAT N - m0 times:
			Choose a link j from U
			REPEAT m times:
			IF rand() < alpha
				Choose a random integer i between 0 and Omega.size
				Establish a link between j and i
				add both i and j to N
			ELSE
				Choose a random float between 0 and some number, depending on that float get an integer i
				i is a node in N
				establish a link between j and i
			END
		END
	***/
	
	for(int k = 0; k < (N - m0); k++)
	{	
		#ifdef GARDEBUG
			fprintf(oFile, "\nMain Loop Iteration: %d\nl  U N\n", k);
			for(int l = 0; l < N; l++)
			{
				fprintf(oFile, "%d  %d %d\n", l, Unconnected[l], Connected[l]);
			}
			fprintf(oFile, "sizeU: %d\n", sizeU);
		#endif
		
		tempIndex = randoInt(0, sizeU - 1);
		
		#ifdef GARDEBUG
			fprintf(oFile, "tempIndex: %d\n", tempIndex);
		#endif
		
		
		tempInt = 0;
		j = 0;
		
		while(tempInt < tempIndex)
		{
			if(Unconnected[j] == 1)
				tempInt++;
			
			j++;
		}
		
		while(Unconnected[j] == 0)
			j++;
			
		#ifdef GARDEBUG
			fprintf(oFile, "Found j: %d\n", j);
		#endif
		
		// Index is our node j, now remove j from U
		Unconnected[j] = 0;
		sizeU -= 1;
		
		for(int l = 0; l < avgDegree; l++)
		{
			if(randoFlat(0, 1) < alpha)
			{
				#ifdef GARDEBUG
					fprintf(oFile, "\nConnected Erdos-Style\n");
				#endif
				
				// Make link via Erdos-Renyi
				// Choose a random node in Omega, make sure it is not tempNode. If it is, choose again
				// Then make the link
				while((i = randoInt(0, N - 1)) == j);
				
				#ifdef GARDEBUG
					fprintf(oFile, "Found i: %d\n", i);
					fprintf(oFile, "Connected %d and %d\n", i, j);
				#endif
				
				addLink(a, i, j, N, degDist);
				
				// Add both i and j to N
				Connected[i] = 1;
				Connected[j] = 1;
				
			} else
			{
				// Make link via Preferential Attachment
				// paFunction takes a degree distribution and N and returns a random index distributed by a function of the degree distribution
				#ifdef GARDEBUG
					fprintf(oFile, "\nConnected PA-Style\n");
				#endif
				if((sizeU == N - 2) && m0 == 1)
					i = 0;
				else	
					while((i = BAConnect(degDist, N, (double)alpha, (double)avgDegree)) == j);
				#ifdef GARDEBUG
					fprintf(oFile, "Found i: %d\nConnected %d and %d\n", i, i, j);
				#endif
				
				addLink(a, i, j, N, degDist);
				Connected[j] = 1;
			}
		}
		#ifdef GARDEBUG
			fprintf(oFile, "\n");
			for(int k = 0; k < N; k++)
			{
				for(int l = 0; l < N; l++)
				{
					fprintf(oFile, "%d ", (int)a[k*N + l]);
				}
				fprintf(oFile, "\n");
			}
		#endif
	}
	
	#ifdef GARDEBUG
		putchar('\n');
		for(int k = 0; k < N; k++)
		{
			for(int l = 0; l < N; l++)
			{
				printf("%d ", (int)a[k*N + l]);
			}
			putchar('\n');
		}
	#endif
	#ifdef GARDEBUG
		for(int l = 0; l < N; l++)
		{
			fprintf(oFile, "%d\n", degDist[l]);
		}
	#endif
	return a;
}

float * genBarabasi(int initial_node_count, int final_node_count, float probability)
{
	float *a;
	int *v;
	int sum_k;

	a = (float*)malloc(final_node_count*final_node_count*sizeof(float));
	v = (int*)malloc(final_node_count*sizeof(int));
		
	for(int i = 0; i < initial_node_count; i++)
		v[i] = 0;

	for(int i = 0; i < initial_node_count; i++)
	{
		
		for(int j = 0; j < i; j++)
		{
			if(randoFlat(0, 1) < probability)
			{
				a[i*final_node_count + j] = 1;
				a[j*final_node_count + i] = 1;
				v[i]++;
				v[j]++;
			} else
			{
				a[i*final_node_count + j] = 0;
				a[j*final_node_count + i] = 0;
			}
		}
		a[i*final_node_count + i] = 0;
	}

	// Now start attaching nodes
	for(int i = initial_node_count; i < final_node_count; i++)
	{
		sum_k = 0;

		// Find Degree Distribution sum
		for(int j = 0; j < i; j++)
		{
			for(int k = 0; k < j; k++)
				sum_k += (int)(a[j*final_node_count + k]);
		}

		for(int j = 0; j < i; j++)
		{
			if(randoFlat(0,1) < ((float)(v[j])/(float)(sum_k)))
			{
				a[i*final_node_count + j] = 1;
				a[j*final_node_count + i] = 1;
				v[i]++;
				v[j]++;
			} else
			{
				a[i*final_node_count + j] = 0;
				a[j*final_node_count + i] = 0;
			}
		}
	}
		

	free(v);
	
	return a;

}

int compDescendingFloat(const void *a, const void *b)
{
	if((*(float*)b - *(float*)a) == 0)
		return 0;
	else if((*(float*)b - *(float*)a) < 0.0)
		return -1;
	else
		return 1;
}

igraph_t *getSynchronizedGraph(float *hDistMatrix, float rLink, int N)
{
	float *tempDistMatrix = (float*)malloc(N*N*sizeof(float));
	float threshold;
	int links = 0;
	int index;
	int temp;
	
	igraph_t *syncGraph = (igraph_t*)malloc(sizeof(igraph_t));
	igraph_vector_t edges;
	
	for(int i = 0; i < N*N; i++)
	{
		// Copy hDistMatrix into tempDistMatrix
		tempDistMatrix[i] = hDistMatrix[i];
		
		if(hDistMatrix[i] != 0.0)
			links++;
	}
	
	links /= 2;
	
	// Now sort the tempDistMatrix
	qsort(tempDistMatrix, N*N, sizeof(float), compDescendingFloat);
	
	// Now get the threshold
	index = (int)2*links*rLink - 1;
	
	if(index < 0)
		index = 0;
	else if(index >= N*N)
		index = N*N - 1;
		
	//printf("\nIndex: %d\nThreshold: %f, %f, %f, %f, %f\n", index, tempDistMatrix[index-2], tempDistMatrix[index-1], tempDistMatrix[index], tempDistMatrix[index+1], tempDistMatrix[index+2]);

	threshold = tempDistMatrix[index];
		
	// Creates an empty undirected graph of N vertices
	igraph_empty(syncGraph, N, 0);
	
	igraph_vector_init(&edges, 2*(index + 1));
	igraph_vector_null(&edges);
	
	temp = 0;
	
	for(int i = 0; i < N; i++)
	{
		for(int j = i; j < N; j++)
		{
			if(hDistMatrix[i*N + j] > threshold)
			{
				VECTOR(edges)[temp] = i;
				temp += 1;
				VECTOR(edges)[temp] = j;
				temp += 1;
			}
		}
	}
	
	igraph_add_edges(syncGraph, &edges, 0);
	
	igraph_vector_destroy(&edges);
	
	free(tempDistMatrix);
	
	return syncGraph;
}

void wrongSyntax()
{
	printf("A reminder of format:\n");
	printf("./kudamoto [Number of Oscillators] [Coupling Constant] [Step Size] [End Time] [Seed] [Frequency Distribution] [Param 1] [Param 2] --erdos_renyi [Probability] [Output File]\n");
	printf("./kudamoto [Number of Oscillators] [Coupling Constant] [Step Size] [End Time] [Seed] [Frequency Distribution] [Param 1] [Param 2] --config_model [Average Degree] [Power] [Output]\n");
	printf("./kudamoto [Number of Oscillators] [Coupling Constant] [Step Size] [End Time] [Seed] [Frequency Distribution] [Param 1] [Param 2] --gardenes [Alpha] [m0] [Average Degree] [Output]\n");
	printf("./kudamoto [Number of Oscillators] [Coupling Constant] [Step Size] [End Time] [Seed] [Frequency Distribution] [Param 1] [Param 2] --barabasi [Initial Node Count] [Erdos-Renyi Probability of Connection] [Output]\n");
	printf("./kudamoto [Number of Oscillators] [Coupling Constant] [Step Size] [End Time] [Seed] [Frequency Distribution] [Param 1] [Param 2] --all_to_all [Output]\n");
	printf("Frequency Distribution: [cauchy/gaussian/flat] Param 1: [Mean/Mean/Min] Param 2: [Spread/Spread/Max]\n");
}
