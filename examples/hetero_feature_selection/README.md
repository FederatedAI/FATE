## Quick Start

We supply standalone and cluster mode of running examples for Feature selection algorithm.

### 1. Run Standalone Version

In standalone mode, you can simply run_feature_selection_standalone.sh, which will run host and guest in the background and print out some infomation, like logfiles path.

> sh run_feature_selection_standalone.sh


###2. Run Cluster Version
In cluster version, you can follow similar steps as decribed in 'Run Standalone Version'

1. Open two terminal(or two session)

2. In each of the two terminal. go to examples/hetero_feature_selection/

3. Start host by 
   > sh run_feature_selection_clustering.sh host $guestid $hostid $jobid

4. Start guest by
   > sh run_feature_selection_clustering.sh guest $guestid $hostid $jobid

Again the job id should be the same for host and guest.


guest_partyid is the role id of guest party, which is set during the installation of FATE. host_partyid has similar
meaning of guest_partyid

### 3. Check log files

The logs are provided in the **logs** directory. A sub-directory will be generated with the name of jobid. All the log files are
listed in this directory. The trained and validation result are shown in feature_selection.log. Feel free to check out each log file
for more training details. 