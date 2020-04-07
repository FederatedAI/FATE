## Quick Start

We will quickly run FTL algorithm of both encrypted version and plain version on standalone mode.


## 1. Run Encrypted FTL

You can start running encrypted FTL through following steps:

1. Open one terminal. Go to the root folder of the FATE project.
2. Go to **examples/hetero_ftl/** folder. 
3. Switch the algorithm to encrypted version. This version is default. 
    * Set <b style="color:red">is_encrypt</b> of **FTLModelParam** to true in both **guest_runtime_conf.json** file and **host_runtime_conf.json** file. These two configuration files are located in **conf/** folder.
    * Set <b style="color:red">enc_ftl</b> of **FTLModelParam** to <b style="color:red">dct_enc_ftl</b> in both **guest_runtime_conf.json** file and **host_runtime_conf.json** file.
        * <b style="color:red">dct_enc_ftl</b> represents decentralized verison of the encrypted FTL algorithm (no arbiter in the loop).
      
Following picture shows an example of parameters in **FTLModelParam** section.

<div style="text-align:center", align=center>
<img src="./images/encrypted_ftl_config.png" />
</div>

4. Under **examples/hetero_ftl/** folder, run **run_ftl_dct_standalone.sh**, which will run host and guest in the background:
    
    > sh run_ftl_dct_standalone.sh 123

    123 is the job id.
    
> You should use a different job id every time you run **run_ftl_dct_standalone.sh** script


### Check log files

You can check logs for the progress or result of a specific job. (almost) All logs are located in **logs/** folder under the FATE project root folder. Logs for the specific job of 123 are located in：

**logs/123/** folder

This folder is named after the job id (e.g., 123 in this case) you specified for running the algorithm.

You normally only need to check logs for host and guest:

* **hetero_ftl_guest.log**, records log information for guest side of running the FTL algorithm. 
  * Guest knows the loss for each iteration. Therefore, you can check the change of loss in this log file.
* **hetero_ftl_host.log**, records log information for host side of running the FTL algorithm. 
  * In our FTL algorithm, it is the host that always triggers predicting process. Therefore, you can check the evaluation result on training or predicting in this log file.

> Note that the progress for running encryption version of the algorithm is relatively slow. You can check the progress by:
>
> tail -f hetero_ftl_host.log

## 2. Run Plain FTL

To run the algorithm, you can follow similar steps as described in "Run Encrypted FTL" section:

1. Open one terminal. Go to the root folder of the FATE project.
2. Go to **examples/hetero_ftl/** folder.
3. Switch the algorithm to plain version. 
    * Set <b style="color:red">is_encrypt</b> of **FTLModelParam** to false in both **guest_runtime_conf.json** file and **host_runtime_conf.json** file. These two configuration files are located in **conf/** folder. When <b style="color:red">is_encrypt</b> is switched to false, <b style="color:red">enc_ftl</b> will not take effect.
 
Following picture shows an example of parameters in **FTLModelParam** section.

<div style="text-align:center", align=center>
<img src="./images/plain_ftl_config.png" />
</div>

4. Run **run_ftl_dct_standalone.sh**, which will run host and guest in the background.
    
    > sh run_ftl_dct_standalone.sh 124

   124 is the job id.  
   
> You should use a different job id every time you run **run_ftl_plain_standalone.sh** script


### Check log files

Go to **logs/124/** folder for checking logs.


 

