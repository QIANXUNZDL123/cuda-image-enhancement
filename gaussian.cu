#include "gaussian.cuh"

constexpr int GAUSSIAN_FILTER_SIZE = 3;

__global__ void k_1D_gaussian_filter(unsigned char *input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = tx * cols + ty;
	
	int offset = GAUSSIAN_FILTER_SIZE / 2;
	int newPixelValue = 0;


	if(threadId >= rows * cols){
		return;
	}

	int conv_kernel[GAUSSIAN_FILTER_SIZE][GAUSSIAN_FILTER_SIZE] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	for (int r = 0; r < mask_dim; r++)
	{
		for (int c = 0; c < mask_dim; c++)
		{
			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1))
			{
				newPixelValue += conv_kernel[r][c] * input[(tx - offset + r) * cols + (ty - offset + c)];
			}
			else
			{
				return;
			}
		}
	}

	input[threadId] = static_cast<uchar>(newPixelValue / 16);
}

__global__ void k_1D_gaussian_filter_shared_mem(unsigned char* input, int rows, int cols, int mask_dim)
{
	__shared__ unsigned char cache[32][33];

	int conv_kernel[GAUSSIAN_FILTER_SIZE][GAUSSIAN_FILTER_SIZE] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = tx * cols + ty;

	int offset = GAUSSIAN_FILTER_SIZE / 2;
	int newPixelValue = 0;

	if (threadId >= rows * cols)
	{
		return;
	}

	unsigned int cy = threadIdx.x;
	unsigned int cx = threadIdx.y;

	cache[cx][cy] = input[threadId]; /*load data shared mem*/
	__syncthreads();

	for (int i = 0; i < mask_dim; i++)
	{
		for (int j = 0; j < mask_dim; j++)
		{ /*travel on conv matrix*/
			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1))
			{
				int x_index = cx - offset + i;
				int y_index = cy - offset + j;
				if (cx == 31 || cx == 0 || cy == 0 || cy == 31)
				{
					newPixelValue += conv_kernel[i][j] * input[(tx - offset + i) * cols + (ty - offset + j)];
				}
				else
				{
					newPixelValue += conv_kernel[i][j] * cache[x_index][y_index];
				}
			}
			else
			{
				return;
			}
		}
	}
	input[threadId] = static_cast<uchar>(newPixelValue / 16);
}

float gaussian_filter_gpu_3D(cv::Mat input_img, cv::Mat *output_img, bool sm)
{
	unsigned char *gpu_input = NULL;

	unsigned char *input = input_img.data;
	unsigned char *output = output_img->data;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;
	unsigned int size = rows * cols * sizeof(unsigned char) * 3;

	const uint mask_dim = 3;

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaHostRegister(input,size,0));
	CHECK_CUDA_ERROR(cudaMalloc((unsigned char **)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));

	if(sm){
		k_3D_gaussian_filter_shared_mem<<<grid, block>>>(gpu_input, rows, cols, mask_dim);
	}
	else{
		k_3D_gaussian_filter<<<grid, block>>>(gpu_input, rows, cols, mask_dim);
	}

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return elapsed;
}
__global__ void k_3D_gaussian_filter(unsigned char *input, int rows, int cols, int mask_dim)
{
	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty ) * 3;

	int offset = GAUSSIAN_FILTER_SIZE / 2;
	int new_red_val = 0;
	int new_green_val = 0;
	int new_blue_val = 0;

	int conv_kernel[GAUSSIAN_FILTER_SIZE][GAUSSIAN_FILTER_SIZE] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	if (threadId >= rows * cols * 3)
	{
		return;
	}
	for (int i = 0; i < mask_dim; i++)
	{
		for (int j = 0; j < mask_dim; j++)
		{
			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1))
			{
				new_red_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + ty - offset + j) * 3];
				new_green_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + ty - offset + j) * 3 + 1];
				new_blue_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + ty - offset + j) * 3 + 2];
			}
			else
			{
				return;
			}
		}
	}
	input[threadId] = static_cast<uchar>(new_red_val / 16);
	input[threadId + 1] = static_cast<uchar>(new_green_val / 16);
	input[threadId + 2] = static_cast<uchar>(new_blue_val / 16);
}
__global__ void k_3D_gaussian_filter_shared_mem(unsigned char *input, int rows, int cols, int mask_dim)
{
	__shared__ unsigned char cache_red[32][32];
	__shared__ unsigned char cache_green[32][32];
	__shared__ unsigned char cache_blue[32][32];

	int conv_kernel[GAUSSIAN_FILTER_SIZE][GAUSSIAN_FILTER_SIZE] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	int ty = blockIdx.x * blockDim.x + threadIdx.x;
	int tx = blockIdx.y * blockDim.y + threadIdx.y;
	int threadId = (tx * cols + ty) * 3;

	int offset = GAUSSIAN_FILTER_SIZE / 2;
	int new_red_val = 0;
	int new_green_val = 0;
	int new_blue_val = 0;

	if (threadId >= rows * cols * 3)
	{
		return;
	}

	unsigned int cy = threadIdx.x;
	unsigned int cx = threadIdx.y;

	cache_red[cx][cy] = input[threadId]; /*load data shared mem*/
	cache_green[cx][cy] = input[threadId + 1]; 
	cache_blue[cx][cy] = input[threadId + 2]; 
	__syncthreads();

	for (int i = 0; i < mask_dim; i++)
	{
		for (int j = 0; j < mask_dim; j++)
		{ /*travel on conv matrix*/
			if ((tx > 0 && tx < rows - 1) && (ty > 0 && ty < cols - 1))
			{
				int x_index = cx - offset + i;
				int y_index = cy - offset + j;
				if (cx == 31 || cx == 0 || cy == 0 || cy == 31)
				{
					new_red_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + (ty - offset + j)) * 3];
					new_green_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + (ty - offset + j)) * 3 + 1];
					new_blue_val += conv_kernel[i][j] * input[((tx - offset + i) * cols + (ty - offset + j)) * 3 + 2];
				}
				else
				{
					new_red_val += conv_kernel[i][j] * cache_red[x_index][y_index];
					new_green_val += conv_kernel[i][j] * cache_green[x_index][y_index];
					new_blue_val += conv_kernel[i][j] * cache_blue[x_index][y_index];
				}
			}
			else
			{
				return;
			}
		}
	}

	input[threadId] = static_cast<uchar>(new_red_val / 16);
	input[threadId + 1] = static_cast<uchar>(new_green_val / 16);
	input[threadId + 2] = static_cast<uchar>(new_blue_val / 16);
}

float gaussian_filter_gpu_1D(cv::Mat input_img, cv::Mat *output_img, bool sm)
{
	unsigned char *gpu_input = NULL;

	unsigned char *input = input_img.data;
	unsigned char *output = output_img->data;

	unsigned int cols = input_img.cols;
	unsigned int rows = input_img.rows;
	unsigned int pixels = cols * rows;
	unsigned int size = pixels * sizeof(unsigned char);

	const uint mask_dim = 3;

	dim3 block(32, 32);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	CHECK_CUDA_ERROR(cudaMalloc((unsigned char **)&gpu_input, size));
	CHECK_CUDA_ERROR(cudaMemcpy(gpu_input, input, size, cudaMemcpyHostToDevice));

	if(sm){
		k_1D_gaussian_filter_shared_mem<<<grid, block>>>(gpu_input, rows, cols, mask_dim);
	}
	else{
		k_1D_gaussian_filter<<<grid, block>>>(gpu_input, rows, cols, mask_dim);
	}

	CHECK_CUDA_ERROR(cudaMemcpy(output, gpu_input, size, cudaMemcpyDeviceToHost));

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float elapsed = 0.0f;
	cudaEventElapsedTime(&elapsed, start, stop);

	cudaFree(gpu_input);
	cudaDeviceReset();
	return elapsed;
}

float gaussian_filter_cpu_3D(cv::Mat input_img, cv::Mat *output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;
	
	float kernel[mask_dim][mask_dim] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	auto start = std::chrono::steady_clock::now();

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int new_red_val = 0;
			int new_green_val = 0;
			int new_blue_val = 0;
			for (int m = 0; m < mask_dim; m++)
			{
				for (int n = 0; n < mask_dim; n++)
				{
					new_red_val += input[(((i + m - 1) * cols + (j + n - 1))) * 3] * kernel[m][n];
					new_green_val += input[((i + m - 1) * cols + (j + n - 1)) * 3 + 1] * kernel[m][n];
					new_blue_val += input[((i + m - 1) * cols + (j + n - 1) )* 3 + 2] * kernel[m][n];
				}
			}
			output[(i * cols + j) * 3] = new_red_val / 16;
			output[(i * cols + j) * 3 + 1] = new_green_val / 16;
			output[(i * cols + j) * 3 + 2] = new_blue_val / 16;
		}
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}
float gaussian_filter_cpu_1D(cv::Mat input_img, cv::Mat *output_img)
{
	int cols = input_img.cols;
	int rows = input_img.rows;

	unsigned char* input = input_img.data;
	unsigned char* output = output_img->data;
	const unsigned short mask_dim = 3;
	float kernel[mask_dim][mask_dim] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
	auto start = std::chrono::steady_clock::now();

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int newPixelValue = 0;
			for (int m = 0; m < mask_dim; m++)
			{
				for (int n = 0; n < mask_dim; n++)
				{
					newPixelValue += input[(i + m - 1) * cols + (j + n - 1)] * kernel[m][n];
				}
			}
			output[i * cols + j] = newPixelValue / 16;
		}
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}

float gaussian_filter_cpu_parallel_1D(cv::Mat input_img, cv::Mat* output_img)
{
	unsigned char *input = input_img.data;
	unsigned char *output = output_img->data;
	int cols = input_img.cols;
	int rows = input_img.rows;
	const unsigned short mask_dim = 3;
	float kernel[mask_dim][mask_dim] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	std::vector<std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < MAX_THREAD_SUPPORT; i++)
	{
		threads.push_back(std::thread([&, i](){
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? cols : stride * (i + 1);

			for (int r = range_start; r < range_end; r++) { /*row loop*/
				for (int c = 0; c < cols; c++) { /*col loop*/
					if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
						int new_pixel_value = 0;
						for (int mr = 0; mr < mask_dim; mr++) { /*matrix row*/
							for (int mc = 0; mc < mask_dim; mc++) { /*matrix col*/
								int r_index = r + mr - 1;
								int c_index = c + mc - 1;
								new_pixel_value += input[r_index * cols + c_index] * kernel[mr][mc];
							}
						}
						output[r * cols + c] = static_cast<unsigned char>(new_pixel_value / 16);
					}
				}
			} }));
	}
	for (std::thread &th : threads)
	{
		th.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}
float gaussian_filter_cpu_parallel_3D(cv::Mat input_img, cv::Mat* output_img)
{
	unsigned char *input = input_img.data;
	unsigned char *output = output_img->data;
	int cols = input_img.cols;
	int rows = input_img.rows;
	const unsigned short mask_dim = 3;
	float kernel[mask_dim][mask_dim] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

	std::vector<std::thread> threads;
	const int MAX_THREAD_SUPPORT = std::thread::hardware_concurrency();

	int stride = rows / MAX_THREAD_SUPPORT;

	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < MAX_THREAD_SUPPORT; i++)
	{
		threads.push_back(std::thread([&, i]()
									  {
			int range_start = stride * i;
			int range_end = (i == MAX_THREAD_SUPPORT - 1) ? cols : stride * (i + 1);

			for (int r = range_start; r < range_end; r++) { /*row loop*/
				for (int c = 0; c < cols; c++) { /*col loop*/
					if (r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
						int new_pixel_value_red = 0;
						int new_pixel_value_green = 0;
						int new_pixel_value_blue = 0;
						for (int mr = 0; mr < mask_dim; mr++) { /*matrix row*/
							for (int mc = 0; mc < mask_dim; mc++) { /*matrix col*/
								int r_index = r + mr - 1;
								int c_index = c + mc - 1;
								new_pixel_value_red += input[(r_index * cols + c_index) * 3] * kernel[mr][mc];
								new_pixel_value_green += input[(r_index * cols + c_index) * 3 + 1] * kernel[mr][mc];
								new_pixel_value_blue += input[(r_index * cols + c_index) * 3 + 2] * kernel[mr][mc];

							}
						}
						output[(r * cols + c) * 3] = static_cast<unsigned char>(new_pixel_value_red / 16);
						output[(r * cols + c) * 3 + 1] = static_cast<unsigned char>(new_pixel_value_green / 16);
						output[(r * cols + c) * 3 + 2] = static_cast<unsigned char>(new_pixel_value_blue / 16);
					}
				}
			} }));
	}
	for (std::thread &th : threads)
	{
		th.join();
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start) / 1000.0f;
	return elapsed.count();
}