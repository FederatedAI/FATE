# FATE Script

**FATE Script** is a language which compiles to Python. FATE Script's Syntax is based on the Python language. Hence, FATE Script is an interpreted, high-level, general-purpose programming language based on FATE project.FATE Script's design philosophy emphasizes the simplification of FATE(Federation AI Technology Enabler) algorithms based on homomorphic encryption and multi-party computation (MPC).

# Base module fate_script

While using FATE Script for programming, module `fate_script` must be specified. Module `fate_script` defines the internal functions for developers. For instance, if developer wants to define a function to convert input data to a datastructure called TensorInEgg, function defined by the developer might be as follow:

    def get_lr_x_table(file_path):
	    ns = str(uuid.uuid1())
	    data = pd.read_csv(file_path).values
	    x = eggroll.table('fata_script_test_data_x_' + str(RuntimeInstance.FEDERATION.role + str	(RuntimeInstance.FEDERATION.job_id)), ns, partition=2, persistent=True)
	    if 'y' in list(data.columns.values):
	    	data_index = 2
	    else:
	    	data_index = 1
	    for i in range(np.shape(data)[0]):
	    	x.put(data[i][0], data[i][data_index])
	    return TensorInEgg(RuntimeInstance.FEDERATION.encrypt_operator, None, x)

This function must be contained in the file:

> `$FATE_install_path/contrib/fate_script/fate_script.py`
   


# Syntax

FATE Script adds Site statement and Encrypt statement based on the Python language. Site statement simplifies the data transfer programming to facilitate the programming of algorithm based on FATE project, and Encrypt statement simplifies the  encryption/decryption of data represented in `Tensor`.

#### Site statement


**site assignment statement:**

    x<<A>> = fate_script.get_lr_x_table('breast_a.csv')
It means party A get a variable x while calling function `get_lr_x_table()` of module fate _script
 .If data needs to transfer between parties, the code is as follow:

	x_g<<G>> = x<<A>>
It means transfer x  from A to G, based on the transfer method definded in FATE project. From another perspective, the code written above can be translated to the Python code as follow:

	if __site__ == "G":
		x=fate_script.get(transfer_variable.x.name, (transfer_variable.x.name + '.' + str(__job_id__)[-14:]), idx = 0) # to G
		print("G finish getting x from A")
	if __site__ == "G":
		x_h =x
The variable `__site__` is an internal variable of `FATE script`. Developers has no need to concern about these internal variables.

**site if statement**

    if<<A>> var == None:
    	print('Test')
The code above can be translated to the Python code as follow:

	if __site__ == 'A':
    	if var == None:
    		print('Test')
Similarly, while statement and for statement have the same grammer.

**site simple statement**

`print<<A>>('Test simple statement')`

The code above can be translated as follow:

     if __site__ == 'A':
    	print('Test simple statement')

#### Encrypt statement

**Encrypt statement**

    [[forward]]<<G>> = forward<<G>>

This statement encrypt the `forward` of party `G` into encrypt data using specifict encrypt library(For instance, Paillier)

Homomorphic calculation is as follow:

    [[agg_wx]]<<G>> = [[forward]]<<G>> + [[forward_h]]<<G>>



**Decrypt statement**

    loss<<A>> = [[loss]]<<G>>

Then party A can obtain the encrypted loss for party G and decrypt it. 


## Multiple parties support
Different from federation algorithm of FATE project, Fate Script supports mulitple parties to finish machine learning algorithm and makes it easier. For instance:

    paillier_pubkey<<A>> = fate_script.init_public_key()
    pub_key<<H>> = paillier_pubkey<<A>>
    pub_key<<D>> = paillier_pubkey<<A>>
    pub_key<<G>> = paillier_pubkey<<A>>
    pub_key<<E>> = paillier_pubkey<<A>>

In this way, party A calls the function `init_public_key()` of module `fate_script` to generate the paillier public key, then party H party D, party G and party E can get the public key from A. It should be noted that FATE Script only support on source but multiple destinations.

## Datastructure

**Tensor datastructure**

FATE Script defines class `Tensor` as the basic datastructure. All variables (for instance, such as input data, forward table and gradient ect) involved in calculation must package into class `TensorInPy` or `TensorInEgg` (Both subclass of `Tensor`).

Input data package into TensorInEgg:

	def get_lr_x_table(file_path):
	    ns = str(uuid.uuid1())
	    data = pd.read_csv(file_path).values
	    x = eggroll.table('fata_script_test_data_x_' + str(RuntimeInstance.FEDERATION.role + str	(RuntimeInstance.FEDERATION.job_id)), ns, partition=2, persistent=True)
	    if 'y' in list(data.columns.values):
	    	data_index = 2
	    else:
	    	data_index = 1
	    for i in range(np.shape(data)[0]):
	    	x.put(data[i][0], data[i][data_index])
	    return TensorInEgg(RuntimeInstance.FEDERATION.encrypt_operator, None, x)


Weight data package into TensorInPy:

	def get_lr_w(file_path):
	    w = np.zeros(get_lr_shape_w(file_path))
	    return TensorInPy(RuntimeInstance.FEDERATION.encrypt_operator, None, w)

**blas operator**

FATE Script's basic operators(matmul, mul, sub, join, hstack ect) are supplied in file:

> `$FATE_install_path/contrib/fate_script/blas/blas.py

Tensor multiplication might be as follow:

    forward =X @ W


## Quick Start

We  supply the standalone and cluster mode of running examples for HeteroLogisticRegression algorithm implemented by FATE Script.


**FATE Script Programming**

FATE Script has implemented the HeteroLogisticRegression algotithm in the file:

> `$FATE_install_path/contrib/fate_script/script/HeteroLR.fml

It should be noted that the FATE Script file should be named by the algorithm name, otherwise it may couse some problems of data transfer.


- **Training procedure**:

		#party A generates public key and distributes it to other party
		paillier_pubkey<<A>> = fate_script.init_public_key()
    	pub_key<<H>> = paillier_pubkey<<A>>
    	pub_key<<D>> = paillier_pubkey<<A>>
    	pub_key<<G>> = paillier_pubkey<<A>>
    	pub_key<<E>> = paillier_pubkey<<A>>
    	
		#party H and G get input data packaged into Tensor		
    	X<<H>> = fate_script.get_lr_x_table("/data/projects/fate/python/examples/data/breast_a.csv")
    	W<<H>>= fate_script.get_lr_w("/data/projects/fate/python/examples/data/breast_a.csv")
    	X<<G>> = fate_script.get_lr_x_table("/data/projects/fate/python/examples/data/breast_b.csv")
    	W<<G>>= fate_script.get_lr_w("/data/projects/fate/python/examples/data/breast_b.csv")
    	Y<<G>> = fate_script.get_lr_y_table("/data/projects/fate/python/examples/data/breast_b.csv")
    	shape_w<<G>> = fate_script.get_lr_shape_w("/data/projects/fate/python/examples/data/breast_b.csv")
    	
    	#party A generates ml_conf and distributes it to other party
    	ml_conf<<A>> = fate_script.init_ml_conf()
    	ml_conf<<H>> = ml_conf<<A>>
    	ml_conf<<D>> = ml_conf<<A>>
    	ml_conf<<G>> = ml_conf<<A>>
    	ml_conf<<E>> = ml_conf<<A>>
    	ml_conf<<R>> = ml_conf<<A>>
    	
    	is_stopped<<A>> = False
    	pre_loss<<A>> = None
    	
		#training iteration
    	for iter_idx in range(ml_conf.iter_num):
	    	for batch_idx in range(ml_conf.batch_num):	
		    	forward<<H>> = X<<H>> @ W<<H>>
				#encrypt forward of party H
		    	[[forward]]<<H>> = forward<<H>>
		    	[[forward_square]]<<H>> = forward<<H>>**2
		    	
		    	[[forward_h]]<<G>> = [[forward]]<<H>>
		    	[[forward_square_h]]<<G>> = [[forward_square]]<<H>>
		    					
		    	forward<<G>> = X<<G>> @ W<<G>>
		    	[[forward]]<<G>> = forward<<G>>
		    	[[forward_square]]<<G>> = forward<<G>>**2
		    	
				#Homomorphic calculation
		    	[[agg_wx]]<<G>> = [[forward]]<<G>> + [[forward_h]]<<G>>
		    	
		    	[[agg_wx_square]]<<G>> = [[forward_square]]<<G>> + [[forward_square_h]]<<G>> + 2 * forward<<G>> * [[forward_h]]<<G>>
		    	
		    	[[fore_gradient]]<<G>> = 0.25 * [[agg_wx]]<<G>> - 0.5 * Y<<G>>
		    	
		    	[[grad_G]]<<G>> = (X<<G>> * [[fore_gradient]]<<G>>).mean()
		    	[[grad_H]]<<H>> = (X<<H>> * [[fore_gradient]]<<G>>).mean()
		    	
		    	[[grad_g]]<<A>> = [[grad_G]]<<G>>
		    	[[grad_h]]<<A>> = [[grad_H]]<<H>>
		    	
				#first do homomorphic calculation of encrypted grad_g and encrypted grad_h then decrypt the result
		    	grad<<A>> = [[grad_g]]<<A>>.hstack([[grad_h]]<<A>>)
		    	
		    	(ml_conf.learning_rate)<<A>> = (ml_conf.learning_rate)<<A>> * 0.999
		    	optim_grad<<A>> = grad<<A>> * ml_conf.learning_rate
		    	
		    	shape_w<<A>> = shape_w<<G>>
		    	
		    	optim_grad_g<<A>> = optim_grad<<A>>.split(shape_w<<A>>[0])[0]
		    	optim_grad_h<<A>> = optim_grad<<A>>.split(shape_w<<A>>[0])[1]
		    	
		    	optim_grad_G<<G>> = optim_grad_g<<A>>
		    	optim_grad_H<<H>> = optim_grad_h<<A>>
		    	
		    	W<<G>> = W<<G>> - optim_grad_G<<G>>
		    	W<<H>> = W<<H>> - optim_grad_H<<H>>
		    	
		    	[[half_ywx]]<<G>> = 0.5 * [[agg_wx]]<<G>> * Y<<G>>
		    	
		    	[[loss]]<<G>> =  ([[half_ywx]]<<G>> * (-1) + [[agg_wx_square]]<<G>> / 8 + np.log(2)).mean()
		    	
				#decrypt statement
		    	loss<<A>> = [[loss]]<<G>>
		    	
		    	if<<A>> pre_loss<<A>> is not None and abs(loss<<A>>.store - pre_loss<<A>>.store) < ml_conf.eps :
		    	is_stopped = True
		    	if<<A>> pre_loss<<A>> is None or abs(loss<<A>>.store - pre_loss<<A>>.store) >= ml_conf.eps :
		    		is_stopped<<A>> = False
		    	pre_loss<<A>> = loss<<A>>
		    	if<<A>> is_stopped<<A>>:
		    		break
		    	is_stopped<<G>> = is_stopped<<A>>
		    	if<<G>> is_stopped<<G>>:
		    		break
		    	is_stopped<<H>> = is_stopped<<A>>
		    	if<<H>> is_stopped<<H>>:
		    		break
		    	if<<A>> iter_idx >= 2:
		    		break
		    	if<<G>> iter_idx >= 2:
		    		break
		    	if<<H>> iter_idx >= 2:
		    		break


- **Prediction procedure**:



	    Z<<H>> = X<<H>> @ W<<H>>
	    Z_h<<G>> = Z<<H>>
	    Z<<G>> = X<<G>> @ W<<G>>
	    Z_agg<<G>> = Z<<G>> + Z_h<<G>>
	    
	    Y_test<<G>> = np.array(list(Y<<G>>.store.collect()))[:,1]
	    Y_pred<<G>> = 1.0 / (1 + (Z_agg<<G>> * -1).map(np.exp))
	    Y_pred<<G>> = np.array(list(Y_pred.store.collect()))[:,1]
	    auc<<G>> = metrics.roc_auc_score(Y_test<<G>>, Y_pred<<G>>)
	    print<<G>>("auc: {}".format(auc))
    


**Configure the route.json**

route.json file path is as follow:

> `$FATE_install_path/contrib/fate_script/conf/route.json

route.json descripe the party id of defferent roles and host IP of cluster which execute the FATE Script job.Using the `HeteroLR.fml` mentioned above as example:

> `$FATE_install_path/contrib/fate_script/script/HeteroLR.fml
		
If developers want to make this piece of code work, route.json must be configured according to this code logic. There is an avaliable version of rout.json as follow:

    {
    	"cluster_a": {
    		"ip":[
    			""
    		]
    		"role": [
    			"H",
    			"D",
				"R"
    		]
    		"party_id":10000,
    		"scene_id":50000
    	},
    	"cluster_b": {
    		"ip":[
    			""
    		]
    		"role": [
    			"A",
    			"G",
    			"E"
    		]
    		"party_id":9999,
    		"scene_id":50000
    	}
    }




This json file descripe that there are two clusters named a and b. Cluster a contains three host whose IP is shown above and three parties running their jobs on it. Its party id is 10000, and scene id is 5000. Cluster a is also the same. The IPs of host in the same cluster must be compatible of the party id of this cluster, otherwise data transfer will not work. 

**Run Standalone Version**

In standalone mode, first, the FATE Script file should be programming correctly and route.json should be configured correctly, then start them on any of the host contained in the route.json through following steps:

> `cd $FATE_install_path/contrib/fate_script/`

> `sh run_fateScript_standalone.sh script/HeteroLR.fml`

The file `HeteroLR.fml` is the FATE Script file written by the developer.

**Run cluster Version**

In cluster mode, developer must specified the host IP which executes the FATE Script job of specific role matches the configuration of route.json. Take the route.json mentioned above as example, role H and role D belongs to the cluster a, hence, FATE Script job of role H , role R and role D must be executed on the hosts of cluster a. Same as the role A ,G and E.
Start the FATE Script job through following steps:

> `cd $FATE_install_path/contrib/fate_script/`

> `sh run_fateScript_cluster.sh $role $job_id script/HeteLR.fml`

More specifically, commands of executing FATE Script job of all parties are as follow:
In any party of the cluster_a, execute any of the following command:

> `sh run_fateScript_cluster.sh H $job_id script/HeteLR.fml`
> 
> `sh run_fateScript_cluster.sh D $job_id script/HeteLR.fml`
> 
> `sh run_fateScript_cluster.sh R $job_id script/HeteLR.fml`

In any party of the cluster_b, execute any of the following command:

> `sh run_fateScript_cluster.sh A $job_id script/HeteLR.fml`
> 
> `sh run_fateScript_cluster.sh G $job_id script/HeteLR.fml`
> 
> `sh run_fateScript_cluster.sh E $job_id script/HeteLR.fml`