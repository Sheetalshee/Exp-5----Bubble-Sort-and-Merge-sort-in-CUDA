# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>

#define BLOCK_SIZE 256

// GPU Bubble Sort Kernel
__global__ void bubbleSortKernel(int *arr, int n, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int left = 2 * idx + (phase % 2);

    if (left + 1 < n) {
        if (arr[left] > arr[left + 1]) {
            int temp = arr[left];
            arr[left] = arr[left + 1];
            arr[left + 1] = temp;
        }
    }
}

// GPU Merge Sort Kernel
__global__ void mergeSortKernel(int *arr, int *temp, int n, int width) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;

    int left = pair * 2 * width;
    if (left >= n) return;

    int mid = left + width;
    int right = left + 2 * width;

    if (mid > n) mid = n;
    if (right > n) right = n;

    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid)
        temp[k++] = arr[i++];

    while (j < right)
        temp[k++] = arr[j++];
}

// CPU Merge Function
void mergeHost(int *arr, int *temp, int left, int mid, int right) {
    int i = left;
    int j = mid;
    int k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i < mid)
        temp[k++] = arr[i++];

    while (j < right)
        temp[k++] = arr[j++];
}

// GPU Bubble Sort
void bubbleSort(int *arr, int n) {
    int *d_arr;

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = (n + 1) / 2;
    int blocks = (threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int phase = 0; phase < n; phase++) {
        bubbleSortKernel<<<blocks, BLOCK_SIZE>>>(d_arr, n, phase);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
}

// GPU Merge Sort
void mergeSort(int *arr, int n) {
    int *d_arr, *d_temp;

    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int *source = d_arr;
    int *destination = d_temp;

    for (int width = 1; width < n; width *= 2) {
        int pairs = (n + 2 * width - 1) / (2 * width);
        int blocks = (pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

        mergeSortKernel<<<blocks, BLOCK_SIZE>>>(source, destination, n, width);

        int *swap = source;
        source = destination;
        destination = swap;
    }

    if (source != d_arr) {
        cudaMemcpy(d_arr, source, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// CPU Bubble Sort
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// CPU Merge Sort
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    int *temp = new int[n];

    for (int width = 1; width < n; width *= 2) {
        for (int left = 0; left < n; left += 2 * width) {
            int mid = std::min(left + width, n);
            int right = std::min(left + 2 * width, n);

            mergeHost(arr, temp, left, mid, right);
        }

        std::copy(temp, temp + n, arr);
    }

    delete[] temp;

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// Check sorted array
int isSorted(int *arr, int n) {
    for (int i = 1; i < n; i++) {
        if (arr[i - 1] > arr[i])
            return 0;
    }
    return 1;
}

// Print Array
void printArray(int *arr, int n) {
    int limit = std::min(n, 20);

    for (int i = 0; i < limit; i++)
        printf("%d ", arr[i]);

    if (n > 20) printf("...");

    printf("\n");
}

// Main Function
int main() {
    int n = 1000;

    int *original = (int*)malloc(n * sizeof(int));
    int *cpuBubble = (int*)malloc(n * sizeof(int));
    int *gpuBubble = (int*)malloc(n * sizeof(int));
    int *cpuMerge = (int*)malloc(n * sizeof(int));
    int *gpuMerge = (int*)malloc(n * sizeof(int));

    srand(42);

    for (int i = 0; i < n; i++)
        original[i] = rand() % 1000;

    std::copy(original, original + n, cpuBubble);
    std::copy(original, original + n, gpuBubble);
    std::copy(original, original + n, cpuMerge);
    std::copy(original, original + n, gpuMerge);

    printf("Original Array:\n");
    printArray(original, n);

    printf("\n--- BUBBLE SORT ---\n");

    bubbleSortCPU(cpuBubble, n);
    printArray(cpuBubble, n);

    bubbleSort(gpuBubble, n);
    printArray(gpuBubble, n);

    printf("CPU Bubble Sort Correct: %s\n",
           isSorted(cpuBubble, n) ? "YES" : "NO");

    printf("GPU Bubble Sort Correct: %s\n",
           isSorted(gpuBubble, n) ? "YES" : "NO");

    printf("\n--- MERGE SORT ---\n");

    mergeSortCPU(cpuMerge, n);
    printArray(cpuMerge, n);

    mergeSort(gpuMerge, n);
    printArray(gpuMerge, n);

    printf("CPU Merge Sort Correct: %s\n",
           isSorted(cpuMerge, n) ? "YES" : "NO");

    printf("GPU Merge Sort Correct: %s\n",
           isSorted(gpuMerge, n) ? "YES" : "NO");

    free(original);
    free(cpuBubble);
    free(gpuBubble);
    free(cpuMerge);
    free(gpuMerge);

    return 0;
}
```

## OUTPUT:
<img width="870" height="422" alt="image" src="https://github.com/user-attachments/assets/c03d336f-e881-458a-bb59-acc6939870f9" />


## RESULT:
The Bubble Sort and Merge Sort algorithms were implemented successfully using CPU and GPU with CUDA. The GPU execution times were 3.333137 ms and 0.374944 ms, respectively. The CPU execution times were displayed as 0.000000 ms.
