# Explainable Federated Learning

Understanding why a model makes a certain prediction and how a feature value contributes to the
predict result are as the same importance as training a accurate model. For a better understanding
of trained federated models, in FATE-1.8 we provides a brand new module 'Model Interpret' which
offers novel model-interpretation algorithms specifically designed for federated machine learning.



# SHAP Based Methods

SHAP(SHapley Additive exPlanations) is the one of the most widely used Model-Agnostic method[1] in 
explainable machine learning. It is developed mainly on two basic concepts: Additive feature attribution methods
and Shapley Values. Here we briefly introduce SHAP and Federated SHAP developed based on SHAP.

Additive feature attribution methods have an explanation model that is a linear 
function of binary variables[2]. Assuming we have an instance with *M* features, 
To explain the predict result *g(z')* 

Below formula shows the definition of additive feature attribution methods:

The Shapley value is a game theory concept that involves fairly 
distributing both gains to several players working in coalition.  

      
                        
                       
## Hetero and Homo Kernel SHAP

How

## Hetero and Homo Tree SHAP

## Reference
[1]https://christophm.github.io/interpretable-ml-book/local-methods.html

[2]A unified approach to interpreting model predictions. Scott M. Lundberg and Su-In Lee. 2017. In Proceedings of the 31st International Conference on Neural Information Processing Systems

[3]