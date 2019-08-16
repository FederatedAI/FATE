## Instructions for Running Toy Example

1. To run the toy example, the following command helps.

     
    python run_toy_example.py ${guest_party_id} ${host_party_id} ${work_mode}
   
    ${guest_party_id}: the party id of role "guest", the role who launch the task.
    ${host_party_id}: the party id of role "host", the coordinator 
    ${work_mode}: 0 for standalone version, 1 for cluster.
    
2. Check logs in screen.

   Once the task starts, some information will be printed out in screen. There are several types of information.
   
   a. Party ID Error or Federation Module Installation Error.
      Once task starts, if it does not print out any information in screen Immediately, and print out errors after several seconds.
      Possibly communication is failed. Maybe the guest_party_id and host_party_id are wrong, or federation module installation is failed. 
      
   b. EggRoll or Federation Error
      
      If jobid is printed out on screen, 
      (1) "job running time exceed" printed out also: checkout federation or host party's EggRoll logs.
      (2) Otherwise, checkout the party guest's EggRoll logs.
      
   c. Boomed! Task Success and logs prints out successfully!.
   
      
   
   

   
