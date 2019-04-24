## Quick Start

We supply standalone and cluster mode of running examples for HomoLogisticRegression algorithm.

### 1. Run Standalone Version

In standalone mode, a host and a guest are invoked. You cdan start the two through following step


1. Open two terminal(or two session)

2. In each of the two terminal. go to examples/homo_logistic_regression/

3. Start host by 
   > sh run_host.sh host_runtime_conf jobid

4. Start guest by
   > sh run_guest.sh guest_runtime_conf jobid

5. Start arbiter by
   > sh run_arbiter.sh arbiter_runtime_conf jobid

jobid is a string which host and guest should have the same jobid, please remember that jobid should be different for different task,
host_runtime_conf or guest_runtime_conf template is under example/homo_logistc_regression/host(guest)_runtime_conf.json.

Or you can simply run_secureboosting_standalone.sh, which will run host and guest in the background and print out some infomation, like logfiles path.

> sh run_logistic_regression_standalone.sh


###2. Run Cluster Version
In cluster version, you can follow similar steps as decribed in 'Run Standalone Version'

1. Open two terminal(or two session)

2. In each of the two terminal. go to examples/homo_logistic_regression/

3. Start host by 
   > sh run_host.sh host_runtime_conf jobid

4. Start guest by
   > sh run_guest.sh guest_runtime_conf jobid

5. Start arbiter by
   > sh run_arbiter.sh arbiter_runtime_conf jobid

Again the job id should be the same for host and guest.

Or you can simply run the following instructions, whicn you don't need to modify the config template automatically.

> sh run_secureboosting_cluster.sh host jobid guest_paryid host_partyid

> sh run_secureboosting_cluster.sh guest jobid guest_partyid host_partyid

guest_partyid is the role id of guest party, which is set during the installation of FATe. host_partyid has similar 
meaning of guest_partyid

### 3. Check log files

The logs are provided in the **logs** directory. A sub-directory will be generated with the name of jobid. All the log files are
listed in this directory. The trained and validation result are shown in workflow.log. Feel free to check out each log file 
for more training details. 