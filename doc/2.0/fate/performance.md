## Performance 
This document mainly introduces the performance comparison of the main algorithms based on FATE-v2.0.0 and FATE-v1.11.4.
  
Testing configuration:
* 2 hosts with same machine configuration
  * System: Centos 7.2 64bit
  * cpu: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz, 32 cores 
  * Memory: 128GB
  * hard disk: 4T
  * Network: Lan, 1Gb
* use 16 cores on each host
* Dataset: 
  * PSI: input data is 100 million, result size is 100 million
  * Other Algorithms: 
    * guest site: 10w * 30 dimensions, host size: 10w * 300 dimensions

| Algorithm               | times on FATE-v2.0.0                     | times on FATE-v1.11.4       |   Improvements     |   
| ------------------------| ---------------------------------------- | --------------------------- | ------------------ |
| PSI                     | 50m54s                                   | 1h32m20s                    |  1.8x+             |                              
| Hetero-SSHE-LR          | 4m54s/epoch                              | 21m03s/epoch                |  4.3x+             |                              
| Hetero-NN               | 52.5s/epoch(based on FedPass protocol)   | 2940s/epoch                 |  56x+              |                              
| Hetero-Coordinated-LR   | 2m16s/epoch                              | 2m41s/epoch                 |  1.2x+             |                              
| Hetero-Feature-Binning  | 1m08s                                    | 1m45s                       |  1.5x+             |                              
