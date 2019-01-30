### Federated Logistic Regression

Logistic Regression(LR) is a widely used statistic model for classification problems. FATE has provided two kinds of federated
LR which are Homogeneous LR(HomoLR) and Heterogeneous LR(HeteroLR). 

We simplified the federation process into three parties.
Party A represent to Guest which holds the label in Hetero mode. Party B represent to Host. Party C, which also known as "Arbiter",
 is a third party that holds the private key for each party and work as a coordinator. 
 
As the name suggested, in HomoLR, the feature spaces in guest and hosts are identity. An optional encryption mode for gradient
are provided for host parties. By doing this, the plainted model is not available for this host any more. 

On the other hands, in Hetero mode, parties should have enough amount of same
sample ids. The combination of each sub-model form a completed model. 

The following two figures can shown the principle of Federated LR.

<div style="text-align:center", align=center>
<img src="./images/HomoLR.png" alt="samples" width="500" height="250" /><br/>
Figure 1： Federated HomoLR Principle</div> 

The HomoLR process can be shown as above figure, Party A and Party B has same structure of model.
In each iteration, each party train model among their own data. After that, they upload their
encrypted(or not, depends on your configuration) gradient to arbiter. The arbiter will aggregate these gradients to form
a federated gradient with which the parties can update their model. Just like the traditional LR, the fitting stop when 
model converge or reach the max iterations. More details is available in this [paper]()

 <div style="text-align:center", align=center>
<img src="./images/HeteroLR.png" alt="samples" width="500" height="250" /><br/>
Figure 2： Federated HeteroLR Principle</div>

The HeteroLR federated parties in a different way. As the figure shown, An intersect process are required before training 
the model. The process is to find out the intersect part among their database. The model are built base on this intersect part.
The intersect process will **not** leakage the sample ids between the parties since the are process in encrypted way. Check out
the paper for more details. 

In the fitting process, party A and party B compute out the elements needed for final gradients. Arbiter aggregate them and compute
out the gradient and then transfer back to each party. Check out the [paper]() for more details.
