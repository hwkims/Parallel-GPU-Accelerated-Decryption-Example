# Parallel GPU-Accelerated Decryption Example

This project demonstrates an optimized, parallelized approach to decrypting data using both CPU and GPU. The decryption process is accelerated on multiple GPUs with asynchronous memory transfers, taking advantage of CUDA streams and OpenMP parallelism. The code is intended to run on a system with multiple GPUs and OpenMP support to achieve high performance for large datasets.

## Features

- **Multiple GPUs Support:** Uses multiple GPUs to process large data chunks in parallel.
- **Asynchronous Memory Transfer:** Leverages CUDA streams for efficient memory transfers between the host (CPU) and device (GPU).
- **OpenMP Parallelism:** CPU-side parallelism with OpenMP for multi-core support.
- **Decryption:** Efficiently decrypts data by permuting and unpermuting 64-bit integers using CUDA kernels.
- **File I/O:** Reads and writes encrypted data from/to a file, ensuring that the decryption process can be resumed if the encrypted file already exists.

## Prerequisites

- **CUDA Toolkit:** The project uses NVIDIA's CUDA for GPU acceleration. Make sure you have the necessary CUDA libraries installed.
- **OpenMP:** OpenMP support is required for parallel execution on the CPU.
- **Multiple GPUs:** The code assumes that your system has multiple GPUs installed.

## Files

- **`helpers.cuh`**: Contains helper functions for CUDA operations, such as memory allocation and error checking.
- **`encryption.cuh`**: Contains functions for encryption and decryption algorithms (`permute64`, `unpermute64`).
- **`main.cpp`**: The main file with the implementation of encryption and decryption, utilizing multiple GPUs.

## Code Overview

### 1. **Encryption on CPU (Serial/Parallel)**

The `encrypt_cpu` function is responsible for encrypting the data on the CPU. It applies the permutation algorithm `permute64` on each entry of the dataset. OpenMP is used to parallelize the loop for multi-core CPUs.

```cpp
void encrypt_cpu(uint64_t * data, uint64_t num_entries, uint64_t num_iters, bool parallel=true) {
    #pragma omp parallel for if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        data[entry] = permute64(entry, num_iters);
}
```

### 2. **GPU Decryption Kernel**

The `decrypt_gpu` kernel runs on the GPU and is responsible for decrypting chunks of the data using the `unpermute64` function. The kernel is designed to be efficient for large datasets, processing multiple data entries in parallel across threads and blocks.

```cpp
__global__ void decrypt_gpu(uint64_t * data, uint64_t num_entries, uint64_t num_iters) {
    const uint64_t thrdID = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = blockDim.x * gridDim.x;

    for (uint64_t entry = thrdID; entry < num_entries; entry += stride)
        data[entry] = unpermute64(data[entry], num_iters);
}
```

### 3. **Multi-GPU Support**

The main code handles data distribution across multiple GPUs. Each GPU processes a portion of the dataset. Data is copied asynchronously between the host (CPU) and each GPU using CUDA streams, which allow for concurrent execution of multiple tasks on different GPUs.

```cpp
for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
    cudaSetDevice(gpu);
    for (uint64_t stream = 0; stream < num_streams; stream++) {
        const uint64_t stream_offset = stream_chunk_size * stream;
        const uint64_t lower = gpu_chunk_size * gpu + stream_offset;
        const uint64_t upper = min(lower + stream_chunk_size, num_entries);
        const uint64_t width = upper - lower;

        cudaMemcpyAsync(data_gpu[gpu] + stream_offset, 
                        data_cpu + lower, 
                        sizeof(uint64_t) * width, 
                        cudaMemcpyHostToDevice, 
                        streams[gpu][stream]);

        decrypt_gpu<<<80*32, 64, 0, streams[gpu][stream]>>>
            (data_gpu[gpu] + stream_offset, width, num_iters);

        cudaMemcpyAsync(data_cpu + lower, 
                        data_gpu[gpu] + stream_offset, 
                        sizeof(uint64_t) * width, 
                        cudaMemcpyDeviceToHost, 
                        streams[gpu][stream]);
    }
}
```

### 4. **Result Verification**

The `check_result_cpu` function compares the decrypted data with the original values to ensure correctness. It checks whether each data entry has been properly decrypted back to its original value.

```cpp
bool check_result_cpu(uint64_t * data, uint64_t num_entries, bool parallel=true) {
    uint64_t counter = 0;
    #pragma omp parallel for reduction(+: counter) if (parallel)
    for (uint64_t entry = 0; entry < num_entries; entry++)
        counter += data[entry] == entry;

    return counter == num_entries;
}
```

### 5. **File I/O**

The program checks if an encrypted file already exists. If the file is found, it reads the encrypted data from the file; otherwise, it encrypts the data and writes it to the file. The `write_encrypted_to_file` and `read_encrypted_from_file` functions handle the file operations.

```cpp
if (!encrypted_file_exists(encrypted_file)) {
    encrypt_cpu(data_cpu, num_entries, num_iters, openmp);
    write_encrypted_to_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
} else {
    read_encrypted_from_file(encrypted_file, data_cpu, sizeof(uint64_t) * num_entries);
}
```

### 6. **Timer**

The `Timer` class is used to measure and print the total execution time for GPU decryption.

```cpp
timer.start();
// GPU decryption code here
timer.stop("total time on GPU");
```

## Execution Flow

1. **Encryption**: If the encrypted file doesn't exist, the program encrypts the data on the CPU and writes it to disk.
2. **Decryption**: The encrypted data is read from the file, and the decryption is performed on multiple GPUs in parallel.
3. **Verification**: After decryption, the program verifies whether the data has been successfully decrypted by comparing it with the original unencrypted values.
4. **Performance**: The code utilizes OpenMP and CUDA streams to maximize performance, both on the CPU (for encryption) and GPU (for decryption).

## How to Compile

To compile the project, use the following command:

```bash
nvcc -o decrypt_example main.cpp -Xcompiler -fopenmp -lcuda
```

This command will compile the project with OpenMP enabled and link the necessary CUDA libraries.

## Running the Program

Execute the compiled program as follows:

```bash
./decrypt_example
```

Ensure that your system has at least the specified number of GPUs and that the necessary CUDA libraries are installed.

## Conclusion

This code provides an efficient multi-GPU decryption solution that can handle large datasets in parallel. By using CUDA for GPU-accelerated decryption and OpenMP for CPU parallelism, it achieves high performance while maintaining the flexibility of working with multiple GPUs and performing asynchronous memory operations.
