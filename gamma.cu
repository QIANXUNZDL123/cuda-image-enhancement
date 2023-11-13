#include "gamma.cuh"

__device__ unsigned char LUT_device[256];


__global__ void k_init_LUT(float gamma) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x);
	LUT_device[threadId] = static_cast<unsigned char>(pow(threadId / 255.0f, gamma) * 255);
}

__global__ void k_3D_gamma_correction(unsigned char* input, int rows, int cols) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * 3;
	
	if (threadId >= rows * cols * 3) {
		return;
	}
	input[threadId] = LUT_device[input[threadId]];
	input[threadId + 1] = LUT_device[input[threadId + 1]];
	input[threadId + 2] = LUT_device[input[threadId + 2]];
}

__global__ void k_3D_gamma_correction_shared_mem(unsigned char* input, int rows, int cols) {
	__shared__ unsigned char cache_LUT[256];

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * 3;

	int threadIdInBlock = (threadIdx.x * blockDim.y) + threadIdx.y;
	if (threadId >= rows * cols * 3) {
		return;
	}

	if (threadIdInBlock < 256) {
		cache_LUT[threadIdInBlock] = LUT_device[threadIdInBlock];
	}
	__syncthreads();
	input[threadId + 2] = cache_LUT[input[threadId + 2]];
	input[threadId + 1] = cache_LUT[input[threadId + 1]];
	input[threadId] = cache_LUT[input[threadId]];
}

float gamma_correction_gpu_3D(cv::Mat input_img, cv::Mat* output_img, float gamma, bool sm) {
	unsigned char* gpu_input = NULL;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;
	unsigned long int size = cols * rows * sizeof(unsigned char) * 3;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	cudaHostAlloc(&input,size,cudaHostAllocDefault);
	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
	if(sm){
		k_init_LUT << <4, 64 >> > (gamma);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		k_3D_gamma_correction_shared_mem << <grid, block >> > (gpu_input, rows, cols);
	}else{
		k_init_LUT << <4, 64 >> > (gamma);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		k_3D_gamma_correction << <grid, block >> > (gpu_input, rows, cols);
	}

	CHECK_CUDA_ERROR(cudaMemcpy(input, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpuElapsedTime = 0;
	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

	cudaFree(gpu_input);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceReset();

	return gpuElapsedTime;
}

float gamma_correction_cpu_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols;
	unsigned int pixels = rows * cols * 3;

	auto start = std::chrono::steady_clock::now();

	unsigned char LUT[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<unsigned char>(pow(i / 255.0f, gamma) * 255);
	}
	for (int i = 0; i < pixels; i++) {
		output[i] = LUT[input[i]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}
float gamma_correction_cpu_parallel_3D(cv::Mat inputImg, cv::Mat* outputImg, float gamma) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	unsigned int rows = inputImg.rows;
	unsigned int cols = inputImg.cols;

	auto start = std::chrono::steady_clock::now();

	std::vector <std::thread> threads; 
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	unsigned char LUT[256] = { 0 };

	for (int i = 0; i < 256; i++) {
		LUT[i] = static_cast<uchar>(pow(i / 255.0f, gamma) * 255);
	}

	for (int i = 0; i < MAX_THREAD_SUPPORT; i++) {
		threads.push_back(std::thread([&,i]() {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? rows : stride * (i + 1);

			for (int x = range_start; x < range_end; x++) {
				for (int y = 0; y < cols; y++) {
					for (int c = 0; c < 3; c++) {
						int index = (x * cols * 3) + (y * 3) + c;
						output[index] = LUT[input[index]];
					}
				}
			}
			}));
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

// float gamma_correction_frames_cpu(cv::Mat* frames, cv::Mat* output_frames,uint frame_count, float gamma) {
// 	float elapsed = 0.0f;
// 	for (int i = 0; i < frame_count; i++) {
// 		elapsed += gamma_correction_cpu(&frames[i], output_frames[i], gamma);
// 	}
// 	return elapsed;
// }
// float gamma_correction_frames_gpu(cv::Mat* input_frames, cv::Mat* output_frames, unsigned int frame_count, float gamma) {
// 	unsigned int cols = input_frames[0].cols;
// 	unsigned int rows = input_frames[0].rows;
// 	unsigned int frame_size = rows * cols * 3 * sizeof(unsigned char);/*size of a frame*/
// 	unsigned int size = frame_size * frame_count; /*size of data to transfer to gpu*/

// 	unsigned char* gpu_input = nullptr;
// 	unsigned char* gpu_output = nullptr;

// 	unsigned char LUT[256] = { 0 };
// 	for (int i = 0; i < 256; i++) {
// 		LUT[i] = static_cast<uchar>(pow(i / 255.0f, gamma) * 255);
// 	}

// 	cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);

// 	cudaEventRecord(start);

// 	CHECK_CUDA_ERROR(cudaMemcpyToSymbolAsync(LUT_constant, LUT, sizeof(unsigned char) * 256, 0));

// 	CHECK_CUDA_ERROR(cudaHostAlloc((void **) & input_frames->data, size, cudaHostAllocDefault));
// 	CHECK_CUDA_ERROR(cudaHostAlloc((void **) & output_frames->data, size, cudaHostAllocDefault));

// 	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_input, size));
// 	CHECK_CUDA_ERROR(cudaMalloc((void**)&gpu_output, size));

// 	for (int i = 0; i < frame_count; i++) {
// 		CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_input + i * frame_size, input_frames[i].data, frame_size, cudaMemcpyHostToDevice, 0));
// 		CHECK_CUDA_ERROR(cudaMemcpyAsync(gpu_output + i * frame_size, output_frames[i].data, frame_size, cudaMemcpyHostToDevice, 0));
// 	}
// 	CHECK_CUDA_ERROR(cudaStreamSynchronize(0));
// 	dim3 block(32, 32);
// 	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

// 	for (int i = 0; i < frame_count; i++) {
// 		shared_gamma_correction << <grid, block >> > (gpu_input + i * frame_size, gpu_output + i * frame_size, rows, cols);
// 	}

// 	for (int i = 0; i < frame_count; i++) {
// 		CHECK_CUDA_ERROR(cudaMemcpyAsync(output_frames[i].data, gpu_output + i * frame_size, frame_size, cudaMemcpyDeviceToHost, 0));
// 	}

// 	cudaEventRecord(stop);
// 	cudaEventSynchronize(stop);
// 	float gpuElapsedTime = 0;
// 	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

// 	cudaFree(gpu_input);
// 	cudaFree(gpu_output);
// 	cudaDeviceReset();
// 	return gpuElapsedTime;
// }