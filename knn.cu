#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;
using namespace cv;

_global_ void knn(uchar4 * input_image, uchar4 * output_image, int k) {
	
	int thread_id = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x;
	int grid_id = block_offset + thread_id;

	output_image[blockIdx.y, grid_id] = input_image[1 + static_cast<int>(round(blockIdx.y / k)), 1 + static_cast<int>(round(grid_id / k))];

}

int main() {

	int k = 4;

	
	uchar4 *h_inputImage, *input_image;
	uchar4 *h_outputImage, *output_image;

	cv::Mat image = cv::imread("C:/2nd Sem/PA/CUDA/Assignment-3/image.jpg");
	
	if (image.empty())
	 {
		std::cerr << "Couldn't open file: " << std::endl;
		exit(1);
	}
	cv::Mat imageInput;
	cv::Mat imageOutput;

	cvtColor(image, imageInput, cv::COLOR_RGB2GRAY);

	int num_Rows = imageInput.rows;
	int num_Cols = imageInput.cols;
	int grid_x = k * num_Rows / 32;
	int grid_y = k * num_Cols / 32;

	h_inputImage = (uchar4 *)imageInput.ptr<unsigned char>(0);
	h_outputImage = (uchar4 *)imageOutput.ptr<unsigned char>(0);

	const int numPixels = num_Rows * num_Cols;

	cudaMalloc((void**)input_image, sizeof(uchar4) * numPixels);
	cudaMalloc((void**)output_image, sizeof(uchar4) * numPixels * k);

	cudaMemcpy(input_image, h_inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);
	dim3 block(32, 1, 1);
	dim3 grid(grid_x, grid_y, 1);

	knn << <grid, block >> > (input_image, output_image, k);

	cudaDeviceSynchronize();

	cudaMemcpy(h_outputImage, output_image, sizeof(uchar4) * numPixels * k, cudaMemcpyDeviceToHost);

	cudaDeviceReset();
}