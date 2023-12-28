# Hetero-SecureBoost Tutorial

In a hetero-federated learning (vertically partitioned data) setting, multiple parties have different feature sets for the same common user samples. Federated learning enables these parties to collaboratively train a model without sharing their actual data. The model is trained locally at each party, and only model updates are shared, not the actual data. 
In FATE-2.0 we introduce our brand new Hetero-NN framework which allows you to 

In this tutorial, we will show you how to run a Hetero-SecureBoost task under FATE-2.0 locally without using a FATE-Pipeline. You can refer to this example for local model experimentation, algorithm modification, and testing, although we do not recommend using it directly in a production environment.

## Setup Hetero-Secureboost Step by Step

To run a Hetero-Secureboost task, several steps are needed:
1. Import required classes in a new python script
2. Prepare tabular data and transform them into fate dataframe
3. Create Launch Function and Run Hetero-Secureboost task
