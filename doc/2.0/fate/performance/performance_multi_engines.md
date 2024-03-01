## Performance 
This document mainly introduces the performance of main algorithm in FATE-v2.0 running of different computing engines
  
Testing configuration:
* 2 hosts with same machine configuration
  * System: Centos 7.2 64bit
  * cpu: Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz, 16 cores 
  * Memory: 32GB
  * hard disk: 1T
  * Network: Lan, 1Gb
* use 16 cores on each host
* Dataset: 
  * PSI-10M: PSI algorithm, input data is 10 million, result size is 10 million
  * PSI-100M: PSI algorithm, input data is 100 million, result size is 100 million
  * Other Algorithms: 
    * guest site: 10w * 30 dimensions, host size: 10w * 300 dimensions
* Job Configuration:
  * computing_partitions: Kp represents `computing_partitions` is K

| Algorithm               | EggRoll-16p  |  EggRoll-32p  | EggRoll-64p  | Spark-Local-16p  | Spark-Local-32p  | Spark-Local-64p  | Standalone-16p  | Standalone-32p  | Standalone-64p  |                 
| ------------------------| -------------| ------------- | ------------ | -----------------| ---------------- | ---------------- | --------------- | --------------- | --------------- |
| PSI-10M                 | 7m43s        | 8m13          | 7m56s        | 7m28s            | 7m13s            | 8m10s            | 11m57s          | 11m57s          | 11m47s          |
| PSI-100M                |              |               |              | 62m              | 69m              | 75m              | 71m             |                 |                 |
| Hetero-SSHE-LR          | 5m17s/epoch  | 5m24s/epoch   | 5m21s/epoch  | 6m4s/epoch       | 6m11s/epoch      | 6m39s/epoch      | 5m41s/epoch     | 5m33s/epoch     | 5m42s/epoch     |
| Hetero-Coordinated-LR   | 2m22s/epoch  | 2m22s/epoch   | 2m23s/epoch  | 2m21s/epoch      | 2m34s/epoch      | 2m28s/epoch      | 2m22s/epoch     | 2m20s/epoch     | 2m23s/epoch     |
| Hetero-SecureBoost      | 2m11s/epoch  | 2m10s/epoch   | 2m9s/epoch   | 3m41s/epoch      | 5m16s/epoch      | 8m23s/epoch      | 2m15s/epoch     | 2m17s/epoch     | 2m15s/tree      |
