
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

#define CHANNELS 3
#define BLURSIZE 9

void ImpError(cudaError_t err)
{
	cout << cudaGetErrorString(err); // << " en " << __FILE__ << __LINE__;
	exit(EXIT_FAILURE);
}

void Imprimir(float* A, int n)
{
	for (int i = 0; i < n; ++i)
		if (i < n) cout << A[i] << " ";
	cout << "\n";
}

__global__
void BlurKernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int it;

	if (col < width && row < height)
	{
		int pixValr, pixValg, pixValb;
		pixValr = pixValg = pixValb = 0;
		int pixels = 0;

		for (int blurRow = -BLURSIZE; blurRow < BLURSIZE + 1; blurRow++)
		{
			for (int blurCol = -BLURSIZE; blurCol < BLURSIZE + 1; blurCol++)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;

				if (curRow > -1 && curRow < height && curCol > -1 && curCol < width)
				{
					int index = curRow * width + curCol;
					pixValr += Pin[3*index];
					pixValg += Pin[3*index + 1];
					pixValb += Pin[3*index + 2];
					pixels++;
				}
			}
		}
		it = row * width + col;
		Pout[3*it] = (unsigned char)(pixValr / pixels);
		Pout[3*it+1] = (unsigned char)(pixValg / pixels);
		Pout[3*it+2] = (unsigned char)(pixValb / pixels);
	}
}

void Blur(unsigned char* Pout, unsigned char* Pin, int width, int height, int n)
{
	int size = n * 3 * sizeof(char);
	unsigned char* d_Pin;
	unsigned char* d_Pout;

	cudaError_t err = cudaSuccess;

	err = cudaMalloc(&d_Pin, size);
	err = cudaMalloc(&d_Pout, size);

	err = cudaMemcpy(d_Pin, Pin, size, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(width/ 32), ceil(height/ 32), 1);
	dim3 dimBlock(32, 32, 1);
	BlurKernel << <dimGrid, dimBlock >> > (d_Pout, d_Pin, width, height);

	err = cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
		ImpError(err);

	cudaFree(d_Pin); cudaFree(d_Pout);
}

int main()
{
	int height, width;
	int n; //height * width
	unsigned char* Pin;
	unsigned char* Pout;

	FileStorage file("salida.txt", FileStorage::WRITE);

	Mat image = imread("lena.tif");
	height = image.rows; width = image.cols;
	cout << "h" << height << "\n";

	n = height * width;

	Pin = new unsigned char[n*3];
	Pout = new unsigned char[n*3];

	Pin = image.data;

	Mat entrada(height, width, CV_8UC3, Pin);

	Blur(Pout, Pin, width, height, n);

	Mat salida(height, width, CV_8UC3, Pout);

	file << "salida" << salida;

	//cout << salida;
	//imshow("Display window", image);
	imshow("Display window", salida);
	waitKey(0);
	imwrite("lena_blur.png", salida);

	//delete Pin;
	//delete Pout;
	//cout << image;
	return 0;
}