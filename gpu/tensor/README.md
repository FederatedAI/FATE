# GPU & FPGA - A Great Heterogeneous Acceleration Engine for Federated Learning

This project is an industrial-level heterogeneous acceleration system to support and speed up federated learning. We've designed and implemented two different heterogeneous acceleration solutions using GPU and FPGA, respectively, that can significantly accelerate the Paillier cryptosystem while maintaining functionality, accuracy and scalability.  

### How to test GPU engine
- Requirements / Recommendations:
    - At least one capable NVIDIA GPU device is required.
        - We would recommend such device with a GPU microarchitecture of or later than Volta, such as Tesla V100 or Tesla V100S, to fully utilize the functional supports in our CUDA code.
    - CentOS with version >= 7.8.2003
        - We haven't tested if our engine works well in other Linux releases, such as Ubuntu and Debian. However, it should work with at most some slight modifications.
    - Python with version >= 3.6.8
        - The latest version of NumPy (1.19.4 as of now) is recommended.
        - You may need to install other essential Python packages.
    - If you would like to compile the CUDA code:
        - gcc version 4.8.5 would suffice. Don't use gcc later than 7 since nvcc doesn't support it.
        - nvcc version 10.0.130 would suffice.
- To test GPU engine functionality
    ```python
    python3 -m paillier_gpu.tests.test_gpu_engine
    ```
- To test GPU engine performance (profiling)
    ```python
    python3 -m paillier_gpu.tests.test_gpu_performance
    ```
- You may switch the RAND_TYPE variable between INT64_TYPE and FLOAT_TYPE in the test file, which is recommended to make sure that both float64 (double) and int64 (long long) types can pass all assertions.

### How to test FPGA engine
- Requirements / Recommendations:
    - At least one capable Xilinx FPGA device, such as Alveo U250, is required.
    - CentOS with version >= 7.8.2003
        - We haven't tested if our engine works well in other Linux releases, such as Ubuntu and Debian. However, it should work with at most some slight modifications.
    - Python with version >= 3.6.8
        - The latest version of NumPy (1.19.4 as of now) is recommended.
        - You may need to install other essential Python packages.
    - GCC with version >= 4.8.5 if you would like to compile the C source code.
    - Superuser privileges are required as we need access sensitive directories.
        - Note that the Python path with and without sudo may differ.
- To test FPGA engine functionality
    ```python
    sudo python3 -m paillier_fpga.tests.test_fpga_engine
    ```
- To test FPGA engine performance (profiling)
    ```python
    sudo python3 -m paillier_fpga.tests.test_fpga_performance
    ```
- You may switch the RAND_TYPE variable between INT64_TYPE and FLOAT_TYPE in the test file, which is recommended to make sure that both float64 (double) and int64 (long long) types can pass all assertions.

### Profiling Information
The profiling result was obtained from a server with the following configuration.  
|Hardware Type|Model|Quantity|Remark|
|-|-|-|-|
|CPU|Intel Xeon Silver 4114 CPU @ 2.20GHz|2|only 1 core is used in profiling|
|GPU|NVIDIA Tesla V100 PCIe 32GB|4|only 1 GPU card is used in profiling|
|FPGA|Xilinx Alveo U250|1||
|Memory|Samsung 16GiB DIMM DDR4 Synchronous 2666 MHz (0.4 ns)|12|192 GB in total|
|Hard Disk|2TB WDC WD20SPZX-60U|1||  

The chart is an overview of the profiling information of our GPU and FPGA engines compared to a CPU implementation under a unified shape of 666*666, where the throughput means the number of operations (instances, either fixed-point numbers or Paillier-encrypted numbers) a device is capable to compute within a second. For matrix multiplication, we consider the number of operations as the number of modular exponentiations we have to compute under a naive O(n^3) algorithm.  
We don't count the memory allocation time as it could take a significant amount of time for I/O-bound operators like those involving modular multiplication instead of modular exponentiation. As a result, we would recommend users to reuse the already-allocated CPU memory space as much as possible in a way similar to register renaming.

|Operator|CPU Throughput|GPU Throughput|GPU Speedup|FPGA Throughput|FPGA Speedup|
|-|-|-|-|-|-|
|fp_encode|62303.97|33611720.05|539.48|7215836.85|115.82|
|fp_decode|567913.21|25958708.28|45.71|583509.90|1.03|
|pi_encrypt|205864.74|24814051.60|120.54|687947.44|3.34|
|pi_gen_obf_seed|444.05|86766.80|195.40|33653.43|75.79|
|pi_obfuscate|60236.27|11101085.43|184.29|2035691.96|33.80|
|pi_decrypt|1590.48|299298.46|188.18|69354.57|43.61|
|fp_mul|228424.79|11480248.47|50.26|1695313.95|7.42|
|pi_add|29759.90|1203071.88|40.43|423378.92|14.23|
|pi_mul|6175.70|1068244.51|172.98|359942.47|58.28|
|pi_matmul|4178.43|620310.10|148.46|150362.36|35.99|
|pi_sum(axis=0)|12865.10|1675271.14|130.22|844531.30|65.65|
|pi_sum(axis=1)|15919.62|4651463.65|292.18|947461.90|59.52|
|pi_sum(axis=None)|10277.66|4677684.56|455.13|877720.61|85.40|