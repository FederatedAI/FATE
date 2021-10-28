Federated Machine Learning
==========================

\[[中文](federatedml_module_zh.md)\]

FederatedML includes implementation of many common machine learning
algorithms on federated learning. All modules are developed in a
decoupling modular approach to enhance scalability. Specifically, we
provide:

1.  Federated Statistic: PSI, Union, Pearson Correlation, etc.
2.  Federated Feature Engineering: Feature Sampling, Feature Binning,
    Feature Selection, etc.
3.  Federated Machine Learning Algorithms: LR, GBDT, DNN,
    TransferLearning, which support Heterogeneous and Homogeneous
    styles.
4.  Model Evaluation: Binary \| Multiclass \| Regression \| Clustering
    Evaluation, Local vs Federated Comparison.
5.  Secure Protocol: Provides multiple security protocols for secure
    multi-party computing and interaction between participants.

![federatedml structure](../../images/federatedml_structure.png){.align-center
width="800px"}

Algorithm List
--------------

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Algorithm                                           Module Name              Description                  Data Input      Data Output                                         Model Input       Model Output
  --------------------------------------------------- ------------------------ ---------------------------- --------------- --------------------------------------------------- ----------------- -------------------
  Reader                                              Reader                   This component loads and     Original Data   Transformed Data                                                      
                                                                               transforms data from storage                                                                                       
                                                                               engine so that data is                                                                                             
                                                                               compatible with FATE                                                                                               
                                                                               computing engine                                                                                                   

  [DataIO](util.md)                                  DataIO                   This component transforms    Table, values   Transformed Table, values are data instance defined                   DataIO Model
                                                                               user-uploaded data into      are raw data.   [here](../python/federatedml/feature/instance.py)                     
                                                                               Instance object(deprecate in                                                                                       
                                                                               FATe-v1.7, use DataTransform                                                                                       
                                                                               instead).                                                                                                          

  [DataTransform](util.md)                           DataTransform            This component transforms    Table, values   Transformed Table, values are data instance defined                   DataTransform Model
                                                                               user-uploaded data into      are raw data.   [here](../python/federatedml/feature/instance.py)                     
                                                                               Instance object.                                                                                                   

  [Intersect](intersect.md)                          Intersection             Compute intersect data set   Table.          Table with only common instance keys.                                 Intersect Model
                                                                               of multiple parties without                                                                                        
                                                                               leakage of difference set                                                                                          
                                                                               information. Mainly used in                                                                                        
                                                                               hetero scenario task.                                                                                              

  [Federated                                          FederatedSample          Federated Sampling data so   Table           Table of sampled data; both random and stratified                     
  Sampling](feature.md#federated-sampling)                                    that its distribution become                 sampling methods are supported.                                       
                                                                               balance in each party.This                                                                                         
                                                                               module supports standalone                                                                                         
                                                                               and federated versions.                                                                                            

  [Feature Scale](feature.md#feature-scale)          FeatureScale             module for feature scaling   Table，values   Transformed Table.                                  Transform factors 
                                                                               and standardization.         are instances.                                                      like min/max,     
                                                                                                                                                                                mean/std.         

  [Hetero Feature                                     Hetero Feature Binning   With binning input data,     Table, values   Transformed Table.                                                    iv/woe, split
  Binning](feature.md#hetero-feature-binning)                                 calculates each column\'s iv are instances.                                                                        points, event
                                                                               and woe and transform data                                                                                         count, non-event
                                                                               according to the binned                                                                                            count etc. of each
                                                                               information.                                                                                                       column.

  [Homo Feature Binning](feature.md)                 Homo Feature Binning     Calculate quantile binning   Table           Transformed Table                                                     Split points of
                                                                               through multiple parties                                                                                           each column

  [OneHot Encoder](feature.md#onehot-encoder)        OneHotEncoder            Transfer a column into       Table, values   Transformed Table with new header.                                    Feature-name
                                                                               one-hot format.              are instances.                                                                        mapping between
                                                                                                                                                                                                  original header and
                                                                                                                                                                                                  new header.

  [Hetero Feature                                     HeteroFeatureSelection   Provide 5 types of filters.  Table           Transformed Table with new header and filtered data If iv filters     Whether each column
  Selection](feature.md#hetero-feature-selection)                             Each filters can select                      instance.                                           used,             is filtered.
                                                                               columns according to user                                                                        hetero\_binning   
                                                                               config                                                                                           model is needed.  

  [Union](union.md)                                  Union                    Combine multiple data tables Tables.         Table with combined values from input Tables.                         
                                                                               into one.                                                                                                          

  [Hetero-LR](logistic_regressionE.md)               HeteroLR                 Build hetero logistic        Table, values   Table, values are instances.                                          Logistic Regression
                                                                               regression model through     are instances                                                                         Model, consists of
                                                                               multiple parties.                                                                                                  model-meta and
                                                                                                                                                                                                  model-param.

  [Local Baseline](local_baseline.md)                LocalBaseline            Wrapper that runs            Table, values   Table, values are instances.                                          
                                                                               sklearn(scikit-learn)        are instances.                                                                        
                                                                               Logistic Regression model                                                                                          
                                                                               with local data.                                                                                                   

  [Hetero-LinR](linear_regression.md)                HeteroLinR               Build hetero linear          Table, values   Table, values are instances.                                          Linear Regression
                                                                               regression model through     are instances.                                                                        Model, consists of
                                                                               multiple parties.                                                                                                  model-meta and
                                                                                                                                                                                                  model-param.

  [Hetero-Poisson](poisson_regression.md)            HeteroPoisson            Build hetero poisson         Table, values   Table, values are instances.                                          Poisson Regression
                                                                               regression model through     are instances.                                                                        Model, consists of
                                                                               multiple parties.                                                                                                  model-meta and
                                                                                                                                                                                                  model-param.

  [Homo-LR](logistic_regression.md)                  HomoLR                   Build homo logistic          Table, values   Table, values are instances.                                          Logistic Regression
                                                                               regression model through     are instances.                                                                        Model, consists of
                                                                               multiple parties.                                                                                                  model-meta and
                                                                                                                                                                                                  model-param.

  [Homo-NN](homo_nn.md)                              HomoNN                   Build homo neural network    Table, values   Table, values are instances.                                          Neural Network
                                                                               model through multiple       are instances.                                                                        Model, consists of
                                                                               parties.                                                                                                           model-meta and
                                                                                                                                                                                                  model-param.

  [Hetero Secure Boosting](ensemble.md)              HeteroSecureBoost        Build hetero secure boosting Table, values   Table, values are instances.                                          SecureBoost Model,
                                                                               model through multiple       are instances.                                                                        consists of
                                                                               parties                                                                                                            model-meta and
                                                                                                                                                                                                  model-param.

  [Hetero Fast Secure Boosting](ensemble.md)         HeteroFastSecureBoost    Build hetero secure boosting Table, values   Table, values are instances.                                          FastSecureBoost
                                                                               model through multiple       are instances.                                                                        Model, consists of
                                                                               parties in layered/mix                                                                                             model-meta and
                                                                               manners.                                                                                                           model-param.

  [Hetero Secure Boost Feature                        SBT Feature Transformer  This component can encode    Table, values   Table, values are instances.                                          SBT Transformer
  Transformer](feature.md#sbt-feature-transformer)                            sample using Hetero SBT leaf are instances.                                                                        Model
                                                                               indices.                                                                                                           

  [Evaluation](evaluation.md)                        Evaluation               Output the model evaluation  Table(s),                                                                             
                                                                               metrics for user.            values are                                                                            
                                                                                                            instances.                                                                            

  [Hetero Pearson](correlation.md)                   HeteroPearson            Calculate hetero correlation Table, values                                                                         
                                                                               of features from different   are instances.                                                                        
                                                                               parties.                                                                                                           

  [Hetero-NN](hetero_nn.md)                          HeteroNN                 Build hetero neural network  Table, values   Table, values are instances.                                          Hetero Neural
                                                                               model.                       are instances.                                                                        Network Model,
                                                                                                                                                                                                  consists of
                                                                                                                                                                                                  model-meta and
                                                                                                                                                                                                  model-param.

  [Homo Secure Boosting](ensemble.md)                HomoSecureBoost          Build homo secure boosting   Table, values   Table, values are instances.                                          SecureBoost Model,
                                                                               model through multiple       are instances.                                                                        consists of
                                                                               parties                                                                                                            model-meta and
                                                                                                                                                                                                  model-param.

  [Homo OneHot                                        HomoOneHotEncoder        Build homo onehot encoder    Table, values   Transformed Table with new header.                                    Feature-name
  Encoder](feature.md#homo-onehot-encoder)                                    model through multiple       are instances.                                                                        mapping between
                                                                               parties.                                                                                                           original header and
                                                                                                                                                                                                  new header.

  [Data Split](data_split.md)                        Data Split               Split one data table into 3  Table, values   3 Tables, values are instance.                                        
                                                                               tables by given ratio or     are instances.                                                                        
                                                                               count                                                                                                              

  [Column Expand](feature.md#column-expand)          Column Expand            Add arbitrary number of      Table, values   Transformed Table with added column(s) and new                        Column Expand Model
                                                                               columns with user-provided   are raw data.   header.                                                               
                                                                               values.                                                                                                            

  [Secure Information Retrieval](sir.md)             Secure Information       Securely retrieves           Table, values   Table, values are instance                                            
                                                      Retrieval                information from host        are instance                                                                          
                                                                               through oblivious transfer                                                                                         

  [Hetero Federated Transfer                          Hetero FTL               Build Hetero FTL Model       Table, values                                                                         Hetero FTL Model
  Learning](hetero_ftl.md)                                                    Between 2 party              are instance                                                                          

  [Hetero KMeans](hetero_kmeans.md)                  Hetero KMeans            Build Hetero KMeans model    Table, values   Table, values are instance; Arbier outputs 2 Tables                   Hetero KMeans Model
                                                                               through multiple parties     are instance                                                                          

  [PSI](psi.md)                                      PSI module               Compute PSI value of         Table, values                                                                         PSI Results
                                                                               features between two table   are instance                                                                          

  [Data Statistics](statistic.md)                    Data Statistics          This component will do some  Table, values   Table                                                                 Statistic Result
                                                                               statistical work on the      are instance                                                                          
                                                                               data, including statistical                                                                                        
                                                                               mean, maximum and minimum,                                                                                         
                                                                               median, etc.                                                                                                       

  [Scorecard](scorecard.md)                          Scorecard                Scale predict score to       Table, values   Table, values are score results                                       
                                                                               credit score by given        are predict                                                                           
                                                                               scaling parameters           score                                                                                 

  [Sample Weight](util.md#sample-weight)             Sample Weight            Assign weight to instances   Table, values   Table, values are weighted instance                                   SampleWeight Model
                                                                               according to user-specified  are instance                                                                          
                                                                               parameters                                                                                                         

  [Feldman Verifiable                                 Feldman Verifiable Sum   This component will sum      Table, values   Table, values are sum results                                         
  Sum](feldman_verifiable_sum.md)                                             multiple privacy values      to sum                                                                                
                                                                               without exposing data                                                                                              

  [Feature                                            Feature Imputation       This component imputes       Table, values   Table, values with missing features filled                            FeatureImputation
  Imputation](feature.md#feature-imputation)                                  missing features using       are Instances                                                                         Model
                                                                               arbitrary methods/values                                                                                           

  [Label Transform](util.md#label-transform)         Label Transform          Replaces label values of     Table, values   Table, values with transformed label values                           LabelTransform
                                                                               input data instances and     are Instances                                                                         Model
                                                                               predict results              or prediction                                                                         
                                                                                                            results                                                                               
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  : Algorithm

Secure Protocol
---------------

-   [Encrypt](secureprotol.md#encrypt)
    -   [Paillier encryption](secureprotol.md#paillier-encryption)
    -   [Affine Homomorphic
        Encryption](secureprotol.md#affine-homomorphic-encryption)
    -   [IterativeAffine Homomorphic
        Encryption](secureprotol.md#iterativeaffine-homomorphic-encryption)
    -   [RSA encryption](secureprotol.md#rst-encryption)
    -   [Fake encryption](secureprotol.md#fake-encryption)
-   [Encode](secureprotol.md#encode)
-   [Diffne Hellman Key
    Exchange](secureprotol.md#diffne-hellman-key-exchange)
-   [SecretShare MPC
    Protocol(SPDZ)](secureprotol.md#secretshare-mpc-protocol-spdz)
-   [Oblivious Transfer](secureprotol.md#oblivious-transfer)
-   [Feldman Verifiable Secret
    Sharing](secureprotol.md#feldman-verifiable-secret-sharing)

Params
------

::: federatedml.param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false

