## Quick Start

We will quickly run intersect module, in both RSA method and simple raw data method.
We support standalone mode and cluster mode of intersection and RSA method is default.
In this module, we implement Party B as role guest, and Party A as role host.

## 1. Run Intersect Standalone

In standalone version, only a host and a guest are involved. You start these two roles through following steps:

1. Go to examples/intersect/
2. sh **run_intersect_standalone.sh**

> sh run_intersect_standalone.sh job_id123

   This will run role host and guest, and print the intersection number after finish intersecting. "job_id123" is the job id.
   
> You should use a different job id every time you run **run_intersect_standalone.sh** script


### Check log files
After the running of intersection finished, You can check the logs for the result. All the logs are located in: **logs/job_id123/** folder.
This folder is named after the job id (e.g., job_id123 in this case) which you specified for running the algorithm.
Normally, you only need to check two logs for running intersection:
**intersect_guest.log** and **intersect_host.log**. These records log information for guest/host side of running the intersection.
After finish intersecting, the results will be save as DType table, which you can find in **logs/job_id123/workflow.log**.

## 2. Run intersect cluster 

 Intersect cluster has two roles which named guest and host.Each role run in one party independently, for instance, Party A and Party B.
To run the federated intersection, you can follow similar steps in each party:

In role guest:
1. Go to examples/intersect/  
2. **sh run_intersect_cluster.sh** 

> sh run_intersect_cluster.sh guest job_id123 9999 10000

This will run role guest. "job_id123" is the job id, "9999" is the party id of role guest, which is set during installation of FATE, while "10000"
is the party id of role host.

In role arbiter:
1. Go to examples/intersect/
2. **sh run_intersect_cluster.sh**

> sh run_intersect_cluster.sh host job_id123 9999 10000

You can check the logs in each party as the standalone above.
>Attention these, the job id should be same of role host and role guest each task and be different every time.