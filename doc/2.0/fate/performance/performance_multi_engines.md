## Performance 
This document mainly introduces the performance of main algorithms in FATE-v2.0 running on different computing engines
  
Testing configuration:
* 2 hosts with same machine configuration
  * System: Centos 7.2 64bit
  * cpu: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz, 16 cores 
  * Memory: 32GB
  * hard disk: 1T
  * Network: Lan, 1Gb
* use 16 cores on each host
* Dataset: 
  * PSI: PSI algorithm, input data is 10 million, result size is 10 million
  * Other Algorithms: 
    * guest site: 10w * 30 dimensions, host size: 10w * 300 dimensions
* Job Configuration:
  * computing_partitions: 16 

| Algorithm               | EggRoll      | Spark-Local | Standalone  |                 
| ------------------------| -------------| ------------| ------------|
| PSI                     | 7m43s        | 7m28s       | 11m57s      |    
| Hetero-SSHE-LR          | 5m17s/epoch  | 6m4s/epoch  | 5m41s/epoch | 
| Hetero-Coordinated-LR   | 2m22s/epoch  | 2m21s/epoch | 2m22s/epoch | 
| Hetero-SecureBoost      | 2m11s/epoch  | 3m41s/epoch | 2m15s/epoch | 
