#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include "EasyBMP.h"

using std::cout;
using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

// GPU-ядро: каждый поток обрабатывает один пиксель
__global__ void median_filter(cudaTextureObject_t tex, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // считываем окно 3x3 через texture memory, граничные пиксели зажимаются через clamp
    unsigned char window[9];
    int k = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            window[k++] = tex2D<unsigned char>(tex, nx, ny);
        }

    // сортировка пузырьком, медиана — элемент [4]
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 9; j++)
            if (window[i] > window[j]) { unsigned char t = window[i]; window[i] = window[j]; window[j] = t; }

    output[y * width + x] = window[4];
}

// CPU-версия фильтра для проверки корректности
void cpuMedianFilter(const vector<unsigned char>& h_input, vector<unsigned char>& h_output, int width, int height) {
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            unsigned char window[9];
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = std::min(std::max(x + dx, 0), width - 1);
                    int ny = std::min(std::max(y + dy, 0), height - 1);
                    window[k++] = h_input[ny * width + nx];
                }
            std::sort(window, window + 9);
            h_output[y * width + x] = window[4];
        }
}

// проверка совпадения результатов CPU и GPU
void verify_result(const vector<unsigned char>& h_cpu, const vector<unsigned char>& h_gpu) {
    for (size_t i = 0; i < h_cpu.size(); i++)
        assert(h_cpu[i] == h_gpu[i]);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./median_filter input.bmp output.bmp\n";
        return -1;
    }

    // загрузка BMP и перевод в grayscale
    BMP input;
    input.ReadFromFile(argv[1]);
    int width = input.TellWidth(), height = input.TellHeight();

    vector<unsigned char> h_input(width * height);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++) {
            RGBApixel p = input.GetPixel(i, j);
            h_input[j * width + i] = (unsigned char)(0.299f*p.Red + 0.587f*p.Green + 0.114f*p.Blue);
        }

    // CPU-фильтрация с замером времени
    vector<unsigned char> h_cpu(width * height);
    auto start_cpu = high_resolution_clock::now();
    cpuMedianFilter(h_input, h_cpu, width, height);
    duration<double> cpu_time = high_resolution_clock::now() - start_cpu;

    // загрузка изображения в texture memory GPU
    cudaChannelFormatDesc cd = cudaCreateChannelDesc<unsigned char>();
    cudaArray* d_array;
    cudaMallocArray(&d_array, &cd, width, height);
    cudaMemcpy2DToArray(d_array, 0, 0, h_input.data(), width, width, height, cudaMemcpyHostToDevice);

    // привязка cudaArray к текстурному объекту
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    // выделение памяти под результат на GPU
    unsigned char* d_output;
    cudaMalloc(&d_output, width * height);

    // каждый блок 16x16 потоков, сетка покрывает всё изображение
    int THREADS = 16;
    int BLOCKS_X = (width  + THREADS - 1) / THREADS;
    int BLOCKS_Y = (height + THREADS - 1) / THREADS;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS_X, BLOCKS_Y);

    // GPU-фильтрация с замером времени
    auto start_gpu = high_resolution_clock::now();
    median_filter<<<blocks, threads>>>(tex, d_output, width, height);
    cudaDeviceSynchronize();
    duration<double> gpu_time = high_resolution_clock::now() - start_gpu;

    // копирование результата с GPU на хост
    vector<unsigned char> h_gpu(width * height);
    cudaMemcpy(h_gpu.data(), d_output, width * height, cudaMemcpyDeviceToHost);

    // проверка корректности результата
    verify_result(h_cpu, h_gpu);

    // вывод результатов
    cout << std::fixed << std::setprecision(6);
    cout << "CPU time: " << cpu_time.count() << " s\n";
    cout << "GPU time: " << gpu_time.count() << " s\n";
    cout << "Speedup:  " << cpu_time.count() / gpu_time.count() << "x\n";

    // сохранение результата в BMP
    BMP out; out.SetSize(width, height); out.SetBitDepth(24);
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++) {
            RGBApixel px; px.Red = px.Green = px.Blue = h_gpu[j*width+i]; px.Alpha = 0;
            out.SetPixel(i, j, px);
        }
    out.WriteToFile(argv[2]);

    // освобождение ресурсов
    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_array);
    cudaFree(d_output);
    return 0;
}
