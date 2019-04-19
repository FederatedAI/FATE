## Quick Start

We will quickly run FTL algorithm of both plain version and encryption version on standalone mode.

## 1. Run Plain Version

In plain version, only a host and a guest are involved. You start the two through following steps:

1. Open one terminal. Go to the root folder of the FATE project.
2. Go to **examples/hetero_ftl/** folder.
3. Run **run_ftl_plain_standalone.sh**, which will run host and guest in the background.
    
    > sh run_ftl_plain_standalone.sh 123

   123 is the job id.  
   
> You should use a different job id every time you run **run_ftl_plain_standalone.sh** script


### Check log files

You can check logs for the progress or result of a specific job. (almost) All logs are located in **logs/** folder under the FATE project root folder. Logs for the specific job of 123 are located in：

**logs/123/** folder

This folder is named after the job id (e.g., 123 in this case) you specified for running the algorithm.

For plain version, you normally only need to check logs for host and guest since arbiter is not in the loop:

* **hetero_ftl_guest.log**, records log information for guest side of running the FTL algorithm. 
  * In plain version, guest knows the loss for each iteration. Therefore, you can check the change of loss in this log file.
* **hetero_ftl_host.log**, records log information for host side of running the FTL algorithm. 
  * In our FTL algorithm, it is the host that always triggers predicting process. Therefore, you can check the evaluation result on training or predicting in this log file.


## 2. Run Encryption Version

In encryption version we have a arbiter in addition to the host and guest. The arbiter is responsible for decrypting messages passed from either host or guest.

To run the algorithm, you can follow similar steps as described in "Run plain Version" section:
1. Open one terminal. Go to the root folder of the FATE project.
2. Go to **examples/hetero_ftl/** folder. 
3. Switch the algorithm to encryption version. That is, setting <b style="color:red">is_encrypt</b> of **FTLModelParam** to true in both **guest_runtime_conf.json** file and **host_runtime_conf.json** file. These two configuration files are located in **conf/** folder.

Following picture shows an example of parameters in **FTLModelParam** section.

<img src="./images/is_encrypt_param.png" />

4. Under **examples/hetero_ftl/** folder, run **run_ftl_enc_standalone.sh**, which will run host, guest and arbiter in the background:
    
    > sh run_ftl_enc_standalone.sh 124

    124 is the job id.
    
> You should use a different job id every time you run **run_ftl_enc_standalone.sh** script

### Check log files

Go to **logs/124/** folder for checking logs

For encryption version, in addition to above two logs (i.e., **hetero_ftl_guest.log** and **hetero_ftl_host.log**), you may also want to check log for arbiter:

* **hetero_ftl_arbiter.log**, records log information for arbiter side of running the FTL algorithm. 
  * In encryption version of FTL algorithm, only arbiter knows the loss for each iteration. Therefore, you can check the change of loss in this log file. 

> Note that the progress for running encryption version of the algorithm is relatively slow. You can check the progress by:
>
> tail -f hetero_ftl_arbiter.log
 

