#include "dependencies.h"

__device__ int dev_histogram[256] = {0};
__device__ float dev_normalized_histogram[256] = {0};
__device__ float dev_cdf[256] = {0};
__device__ int dev_equalization_values[256] = {0};

/*color gpu variables*/

__device__ int dev_histogram_red[256] = {0};
__device__ float dev_normalized_histogram_red[256] = {0};
__device__ float dev_cdf_red[256] = {0};
__device__ int dev_equalization_values_red[256] = {0};

__device__ int dev_histogram_green[256] = {0};
__device__ float dev_normalized_histogram_green[256] = {0};
__device__ float dev_cdf_green[256] = {0};
__device__ int dev_equalization_values_green[256] = {0};

__device__ int dev_histogram_blue[256] = {0};
__device__ float dev_normalized_histogram_blue[256] = {0};
__device__ float dev_cdf_blue[256] = {0};
__device__ int dev_equalization_values_blue[256] = {0};

__global__ void k_1D_extract_histogram(unsigned char* input, int data_size, int thread_load) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;
	
	if (threadId >= data_size) {
		return;
	}

	for(int i = 0 ; i < thread_load; i++){
		atomicAdd(&dev_histogram[input[threadId + i]], 1);
	}
}

__global__ void k_1D_extract_histogram_shared_mem(unsigned char* input, int data_size, int thread_load) {
	__shared__ unsigned int cache[256];

	int thread_id_in_block = (threadIdx.x * blockDim.y) + threadIdx.y;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if (threadId >= data_size) {
		return;
	}

	if (thread_id_in_block < 256) {
		cache[thread_id_in_block] = 0;
	}
	__syncthreads();

	for(int i = 0 ; i < thread_load; i++){
		int idx = threadId + i;
		atomicAdd(&cache[(input[idx])], 1);
	}
	__syncthreads();

	if (thread_id_in_block < 256) {
		atomicAdd(&dev_histogram[thread_id_in_block], cache[thread_id_in_block]);
	}
}

__global__ void k_1D_normalize_cdf_equalization(int pixels) {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;

	dev_normalized_histogram[threadId] = dev_histogram[threadId] / (float)(pixels);
	__syncthreads();

	for (int i = 0; i <= threadId; i++) {
		sum += dev_normalized_histogram[i];
	}
	dev_cdf[threadId] = sum;
	dev_equalization_values[threadId] = int((dev_cdf[threadId] * 255.0f) + 0.5f);
}

__global__ void k_1D_normalize_cdf_equalization_shared_mem(int pixels) {
	__shared__ float cache_normalized_histogram[256];
	__shared__ float cache_cdf[256];

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	cache_normalized_histogram[threadId] = dev_histogram[threadId] / (float)(pixels);
	__syncthreads();
	float sum = 0.0f;
	for (int i = 0; i <= threadId; i++) {
		sum += cache_normalized_histogram[i];
	}
	cache_cdf[threadId] = sum;
	dev_equalization_values[threadId] = int((cache_cdf[threadId] * 255.0f) + 0.5f);
}

__global__ void k_1D_histogram_equalization(unsigned char* input, int pixels, int thread_load) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if (threadId >= pixels) {
		return;
	}

	for(int i = 0; i < thread_load; i++){
		input[threadId + i] = static_cast<uchar>(dev_equalization_values[input[threadId + i]]);
	}
}

__global__ void k_1D_histogram_equalization_shared_mem(unsigned char* input, int pixels, int thread_load) { /*load the cache before threadId control*/
	__shared__ int cache_equalization_values[256];
	
	int thread_id_in_block = (threadIdx.x * blockDim.y) + threadIdx.y;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if(thread_id_in_block < 256){
		cache_equalization_values[thread_id_in_block] = dev_equalization_values[thread_id_in_block];
	}

	if (threadId >= pixels) {
		return;
	}

	__syncthreads();
	for(int i = 0 ; i < thread_load; i++){
		input[threadId + i] = static_cast<uchar>(cache_equalization_values[input[threadId + i]]);
	}
}


__global__ void k_3D_extract_histogram(unsigned char* input, int pixels, int thread_load) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load; 
	
	if (threadId >= pixels) {
		return;
	}

	for(int i = 0; i < thread_load; i++){
		int idx = threadId + i;
		switch (idx % 3)
		{
		case 0:
			atomicAdd(&dev_histogram_red[input[idx]], 1);
			break;
		case 1:
			atomicAdd(&dev_histogram_green[input[idx]], 1);
			break;
		case 2:
			atomicAdd(&dev_histogram_blue[input[idx]], 1);
			break;
		default:
			break;
		}
	}
}

__global__ void k_3D_extract_histogram_shared_mem(unsigned char* input, int channels, int thread_load) {
	__shared__ unsigned int cache_histogram_red[256];
	__shared__ unsigned int cache_histogram_green[256];
	__shared__ unsigned int cache_histogram_blue[256];

	int threadIdInBlock = (threadIdx.x * blockDim.y) + threadIdx.y;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if (threadIdInBlock < 256) {
		cache_histogram_red[threadIdInBlock] = 0;
		cache_histogram_green[threadIdInBlock] = 0;
		cache_histogram_blue[threadIdInBlock] = 0;
	}
	__syncthreads();
	if (threadId >= channels) {
		return;
	}

	for(int i = 0 ; i < thread_load ; i++){
		int idx = threadId + i;
		switch (idx % 3)
		{
		case 0:
			atomicAdd(&cache_histogram_red[(input[idx])], 1);
			break;
		case 1:
			atomicAdd(&cache_histogram_green[(input[idx])], 1);
			break;
		case 2:
			atomicAdd(&cache_histogram_blue[(input[idx])], 1);
		default:
			break;
		}
	}
	__syncthreads();

	if (threadIdInBlock < 256) {
		atomicAdd(&dev_histogram_red[threadIdInBlock], cache_histogram_red[threadIdInBlock]);
		atomicAdd(&dev_histogram_green[threadIdInBlock], cache_histogram_green[threadIdInBlock]);
		atomicAdd(&dev_histogram_blue[threadIdInBlock], cache_histogram_blue[threadIdInBlock]);
	}
}

__global__ void k_3D_normalize_cdf_equalization(int pixels) { /*1,256*/
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	dev_normalized_histogram_red[threadId] = dev_histogram_red[threadId] / (float)(pixels);
	dev_normalized_histogram_green[threadId] = dev_histogram_green[threadId] / (float)(pixels);
	dev_normalized_histogram_blue[threadId] = dev_histogram_blue[threadId] / (float)(pixels);
	__syncthreads();

	float sum_red = 0.0f, sum_green = 0.0f, sum_blue = 0.0f;
	for (int i = 0; i <= threadId; i++) {
		sum_red += dev_normalized_histogram_red[i];
		sum_green += dev_normalized_histogram_green[i];
		sum_blue += dev_normalized_histogram_blue[i];
	}
	dev_cdf_red[threadId] = sum_red;
	dev_cdf_green[threadId] = sum_green;
	dev_cdf_blue[threadId] = sum_blue;
	__syncthreads();

	dev_equalization_values_red[threadId] = int((dev_cdf_red[threadId] * 255.0f) + 0.5f);
	dev_equalization_values_green[threadId] = int((dev_cdf_green[threadId] * 255.0f) + 0.5f);
	dev_equalization_values_blue[threadId] = int((dev_cdf_blue[threadId] * 255.0f) + 0.5f);
}

__global__ void k_3D_normalize_cdf_equalization_shared_mem(int pixels) { /*1,256*/
	__shared__ float cache_normalized_histogram_red[256];
	__shared__ float cache_normalized_histogram_green[256];
	__shared__ float cache_normalized_histogram_blue[256];

	__shared__ float cache_cdf_red[256];
	__shared__ float cache_cdf_green[256];
	__shared__ float cache_cdf_blue[256];

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	cache_normalized_histogram_red[threadId] = dev_histogram_red[threadId] / (float)(pixels);
	cache_normalized_histogram_green[threadId] = dev_histogram_green[threadId] / (float)(pixels);
	cache_normalized_histogram_blue[threadId] = dev_histogram_blue[threadId] / (float)(pixels);
	__syncthreads();

	float sum_red = 0.0f, sum_green = 0.0f, sum_blue = 0.0f;
	for (int i = 0; i <= threadId; i++) {
		sum_red += cache_normalized_histogram_red[i];
		sum_green += cache_normalized_histogram_green[i];
		sum_blue += cache_normalized_histogram_blue[i];
	}
	cache_cdf_red[threadId] = sum_red;
	cache_cdf_green[threadId] = sum_green;
	cache_cdf_blue[threadId] = sum_blue;

	dev_equalization_values_red[threadId] = int((cache_cdf_red[threadId] * 255.0f) + 0.5f);
	dev_equalization_values_green[threadId] = int((cache_cdf_green[threadId] * 255.0f) + 0.5f);
	dev_equalization_values_blue[threadId] = int((cache_cdf_blue[threadId] * 255.0f) + 0.5f);
}

__global__ void k_3D_histogram_equalization(unsigned char* input, int channels, int thread_load) {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if (threadId >= channels) {
		return;
	}

	for(int i = 0; i < thread_load; i++){
		int idx = threadId + i;
		switch (idx % 3)
		{
		case 0:
			input[idx] = static_cast<uchar>(dev_equalization_values_red[input[idx]]);
			break;
		case 1:
			input[idx] = static_cast<uchar>(dev_equalization_values_green[input[idx]]);
			break;
		case 2:
			input[idx] = static_cast<uchar>(dev_equalization_values_blue[input[idx]]);
			break;
		default:
			break;
		}
	}
}

__global__ void k_3D_histogram_equalization_shared_mem(unsigned char* input, int channels, int thread_load) {
	__shared__ int cache_equalization_values_red[256 + 2];
	__shared__ int cache_equalization_values_green[256 + 2];
	__shared__ int cache_equalization_values_blue[256 + 2];
	
	int thread_id_in_block = (threadIdx.x * blockDim.y) + threadIdx.y;

	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = (blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x) * thread_load;

	if (thread_id_in_block < 256) {
		cache_equalization_values_red[thread_id_in_block + 1] = dev_equalization_values_red[thread_id_in_block];
		cache_equalization_values_green[thread_id_in_block + 1] = dev_equalization_values_green[thread_id_in_block];
		cache_equalization_values_blue[thread_id_in_block + 1] = dev_equalization_values_blue[thread_id_in_block];
	}

	if (threadId >= channels) {
		return;
	}
	__syncthreads();
	for(int i = 0 ; i < thread_load; i++){
		int idx = threadId + i;
		switch (idx % 3)
		{
		case 0:
			input[idx] = static_cast<uchar>(cache_equalization_values_red[input[idx] + 1]);
			break;
		case 1:
			input[idx] = static_cast<uchar>(cache_equalization_values_green[input[idx] + 1]);
			break;
		case 2:
			input[idx] = static_cast<uchar>(cache_equalization_values_blue[input[idx] + 1]);
			break;
		
		default:
			break;
		}
	}
}

float histogram_equalization_gpu_3D(cv::Mat input_img, cv::Mat* output_img, bool sm) {
	unsigned char* gpu_input = nullptr;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	unsigned int thread_load = 9;
	unsigned int cols = input_img.cols;
	unsigned int true_cols = cols * 3;
	unsigned int rows = input_img.rows;

	unsigned int pixels = cols * rows;
	unsigned int channels = pixels * 3;
	
	unsigned long int size = cols * rows * sizeof(unsigned char) * 3;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid((true_cols / thread_load + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	if(sm){
		k_3D_extract_histogram_shared_mem << <grid, block >> > (gpu_input, channels, thread_load);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_3D_normalize_cdf_equalization_shared_mem << <1, 256 >> > (pixels);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_3D_histogram_equalization_shared_mem << <grid, block >> > (gpu_input, channels, thread_load);
	}else{
		k_3D_extract_histogram << <grid, block >> > (gpu_input, channels, thread_load);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_3D_normalize_cdf_equalization<< <1, 256 >> > (pixels);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_3D_histogram_equalization<< <grid, block >> > (gpu_input, channels, thread_load);
	}
	
	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpuElapsedTime = 0;
	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return gpuElapsedTime;
}

float histogram_equalization_gpu_1D(cv::Mat input_img, cv::Mat* output_img, bool sm) {
	unsigned char* gpu_input = nullptr;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	int thread_load = 9;
	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;

	unsigned int pixels = input_img.cols * input_img.rows;
	unsigned long int data_size = pixels * sizeof(unsigned char);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char**)&gpu_input, data_size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, data_size, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid((cols / thread_load + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	if(sm){
		k_1D_extract_histogram_shared_mem << <grid, block >> > (gpu_input, pixels, thread_load);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_1D_normalize_cdf_equalization_shared_mem << <1, 256 >> > (pixels);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_1D_histogram_equalization_shared_mem << <grid, block >> > (gpu_input, pixels, thread_load);
	}else{
		k_1D_extract_histogram << <grid, block >> > (gpu_input, pixels, thread_load);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_1D_normalize_cdf_equalization << <1, 256 >> > (pixels);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		k_1D_histogram_equalization<< <grid, block >> > (gpu_input, pixels, thread_load);
	}
	
	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, data_size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float gpuElapsedTime = 0;
	cudaEventElapsedTime(&gpuElapsedTime, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return gpuElapsedTime;
}
float histogram_equalization_cpu_1D(cv::Mat inputImg, cv::Mat* outputImg) {
	unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	int histogram[256] = { 0 };
	float cdf[256] = { 0 };
	float normalizedHistogram[256] = { 0 };
	int equalization[256] = { 0 };
	
	int pixels = inputImg.cols * inputImg.rows;

	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < pixels; i++) {
		histogram[input[i]]++;
	}

	for (int i = 0; i < 256; i++) { 
		normalizedHistogram[i] = (histogram[i] / (float)pixels);
	}

	cdf[0] = normalizedHistogram[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + normalizedHistogram[i];
	}

	for (int i = 0; i < 256; i++) {
		equalization[i] = int((cdf[i] * 255.0f) + 0.5f);
	}

	for (int i = 0; i < pixels; i++) {
		output[i] = equalization[input[i]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float histogram_equalization_cpu_3D(cv::Mat input_img, cv::Mat* output_img) {
	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;

	int histogram_red[256] = { 0 };
	int histogram_green[256] = { 0 };
	int histogram_blue[256] = { 0 };

	float normalize_histogram_red[256] = { 0 };
	float normalize_histogram_green[256] = { 0 };
	float normalize_histogram_blue[256] = { 0 };

	float cdf_red[256] = { 0 };
	float cdf_green[256] = { 0 };
	float cdf_blue[256] = { 0 };

	int equalization_red[256] = { 0 };
	int equalization_green[256] = { 0 };
	int equalization_blue[256] = { 0 };

	int pixels = input_img.cols * input_img.rows;
	int size = pixels * 3 * sizeof(unsigned char);

	auto start = std::chrono::steady_clock::now();

	for (int i = 0; i < pixels; i++) { /*Calculating histogram of input image*/
		histogram_red[input[i * 3]]++;
		histogram_green[input[i * 3 + 1]]++;
		histogram_blue[input[i * 3 + 2]]++;
	}
	for (int i = 0; i < 256; i++) { /*Calculating normalized histogram (better calculation speed)*/
		normalize_histogram_red[i] = (histogram_red[i] / (float)pixels);
		normalize_histogram_green[i] = (histogram_green[i] / (float)pixels);
		normalize_histogram_blue[i] = (histogram_blue[i] / (float)pixels);
	}

	cdf_red[0] = normalize_histogram_red[0];
	cdf_green[0] = normalize_histogram_green[0];
	cdf_blue[0] = normalize_histogram_blue[0];

	for (int i = 1; i < 256; i++) { /*Generating CDF array*/
		cdf_red[i] = cdf_red[i - 1] + normalize_histogram_red[i];
		cdf_green[i] = cdf_green[i - 1] + normalize_histogram_green[i];
		cdf_blue[i] = cdf_blue[i - 1] + normalize_histogram_blue[i];
	}

	for (int i = 0; i < 256; i++) { /*Generating new pixel intensity values then assign them*/
		equalization_red[i] = int((cdf_red[i] * 255.0f) + 0.5f);
		equalization_green[i] = int((cdf_green[i] * 255.0f) + 0.5f);
		equalization_blue[i] = int((cdf_blue[i] * 255.0f) + 0.5f);
	}
	for (int i = 0; i < pixels; i++) {
		output[i * 3] = equalization_red[input[i * 3]];
		output[i * 3 + 1] = equalization_green[input[i * 3 + 1]];
		output[i * 3 + 2] = equalization_blue[input[i * 3 + 2]];
	}

	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float histogram_equalization_cpu_parallel_1D(cv::Mat inputImg, cv::Mat* outputImg) {
	const unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	const unsigned int rows = inputImg.rows;
	const unsigned int cols = inputImg.cols;

	int histogram[256] = { 0 };
	float normalizedHistogram[256] = { 0 };
	float cdf[256] = { 0 };
	int equalization[256] = { 0 };
	int pixels = cols * rows;

	std::vector <std::thread> threads;
	std::mutex mtx;
	std::condition_variable cv;

	const int MAX_THREAD_SUPPORT = 12;
	const int stride = rows / MAX_THREAD_SUPPORT;
	const int stride_for_256 = 256 / MAX_THREAD_SUPPORT;

	int step1_count = 0;
	int step2_count = 0;
	int step3_count = 0;
	int step4_count = 0;
	auto start = std::chrono::steady_clock::now();

	for (int id = 0; id < MAX_THREAD_SUPPORT; id++) {
		threads.push_back(std::thread([&,id] () {
			int range_start = stride * id;
			int range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);

			int t_histogram[256] = {0};

			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					{
						t_histogram[input[r * cols + c]]++;
					}
				}
			}
			
			{
				std::unique_lock<std::mutex> lck(mtx);
				for(int i = 0 ; i < 256 ; i++){
					histogram[i] += t_histogram[i];
				}
			}
			
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step1_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride_for_256 * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? 256 : stride_for_256 * (id + 1);

			for (int i = range_start; i < range_end; i++) {
				normalizedHistogram[i] = histogram[i] / (float)pixels;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step2_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}
			cdf[0] = normalizedHistogram[0];

			for (int i = range_start; i < range_end; i++) {
				if(i == 0)
					continue;
				float sum = 0.0f;
				for (int j = 0; j <= i; j++) {
					sum += normalizedHistogram[j];
				}
				cdf[i] = sum;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step3_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			for (int i = range_start; i < range_end; i++) {
				equalization[i] = int((cdf[i] * 255.0f) + 0.5f);
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step4_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);
			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					int index = r * cols + c;
					output[index] = equalization[input[index]];
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

float histogram_equalization_cpu_parallel_3D(cv::Mat inputImg, cv::Mat* outputImg) {
	const unsigned char* input = inputImg.data;
	unsigned char* output = outputImg->data;

	const unsigned int rows = inputImg.rows;
	const unsigned int cols = inputImg.cols;

	int histogram_red[256] = { 0 };
	int histogram_green[256] = { 0 };
	int histogram_blue[256] = { 0 };

	float normalize_histogram_red[256] = { 0 };
	float normalize_histogram_green[256] = { 0 };
	float normalize_histogram_blue[256] = { 0 };

	float cdf_red[256] = { 0 };
	float cdf_green[256] = { 0 };
	float cdf_blue[256] = { 0 };

	int equalization_red[256] = { 0 };
	int equalization_green[256] = { 0 };
	int equalization_blue[256] = { 0 };
	int pixels = cols * rows;

	std::vector <std::thread> threads;
	std::mutex mtx;
	std::condition_variable cv;

	const int MAX_THREAD_SUPPORT = 12;
	const int stride = rows / MAX_THREAD_SUPPORT;
	const int stride_for_256 = 256 / MAX_THREAD_SUPPORT;

	int step1_count = 0;
	int step2_count = 0;
	int step3_count = 0;
	int step4_count = 0;
	auto start = std::chrono::steady_clock::now();

	for (int id = 0; id < MAX_THREAD_SUPPORT; id++) {
		threads.push_back(std::thread([&,id] () {
			int range_start = stride * id;
			int range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);

			int local_histogram_red[256] = {0};
			int local_histogram_green[256] = {0};
			int local_histogram_blue[256] = {0};

			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					{
						int index = (r * cols + c) * 3;
						local_histogram_red[input[index]]++;
						local_histogram_green[input[index + 1]]++;
						local_histogram_blue[input[index + 2]]++;
					}
				}
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				for(int i = 0 ; i < 256 ; i++){
					histogram_red[i] += local_histogram_red[i];
					histogram_green[i] += local_histogram_green[i];
					histogram_blue[i] += local_histogram_blue[i];
				}
			}
			
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step1_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride_for_256 * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? 256 : stride_for_256 * (id + 1);

			for (int i = range_start; i < range_end; i++) {
				normalize_histogram_red[i] = histogram_red[i] / (float)pixels;
				normalize_histogram_green[i] = histogram_green[i] / (float)pixels;
				normalize_histogram_blue[i] = histogram_blue[i] / (float)pixels;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step2_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}
			cdf_red[0] = normalize_histogram_red[0];
			cdf_green[0] = normalize_histogram_green[0];
			cdf_blue[0] = normalize_histogram_blue[0];

			for (int i = range_start; i < range_end; i++) {
				float sum_red = 0;
				float sum_green = 0;
				float sum_blue = 0;
				for (int j = 0; j <= i; j++) {
					sum_red += normalize_histogram_red[j];
					sum_green += normalize_histogram_green[j];
					sum_blue += normalize_histogram_blue[j];
				}
				cdf_red[i] = sum_red;
				cdf_green[i] = sum_green;
				cdf_blue[i] = sum_blue;
			}
			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step3_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			for (int i = range_start; i < range_end; i++) {
				equalization_red[i] = int((cdf_red[i] * 255.0f) + 0.5f);
				equalization_green[i] = int((cdf_green[i] * 255.0f) + 0.5f);
				equalization_blue[i] = int((cdf_blue[i] * 255.0f) + 0.5f);
			}

			{
				std::unique_lock<std::mutex> lck(mtx);
				if (++step4_count == MAX_THREAD_SUPPORT) {
					cv.notify_all();
				}
				else {
					cv.wait(lck);
				}
			}

			range_start = stride * id;
			range_end = (id == MAX_THREAD_SUPPORT - 1) ? rows : stride * (id + 1);
			for (int r = range_start; r < range_end; r++) {
				for (int c = 0; c < cols; c++) {
					int index = (r * cols + c) * 3;
					output[index] = equalization_red[input[index]];
					output[index + 1] = equalization_green[input[index + 1]];
					output[index + 2] = equalization_blue[input[index + 2]];
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