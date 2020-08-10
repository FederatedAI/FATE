### Population Stability Index (PSI)

#### Introduction
Population stability index (PSI) is a metric to measure how much a feature
has shifted in distribution between two sample sets. Usually, PSI is used to measure the stability of
models or qualities of features. In FATE, PSI module is used to compute PSI value of a feature between
two tables.

Given two data columns: expect & actual, PSI will be computed by the following steps:
* expect column and actual column conduct quantile feature binning
* compute interval percentage, which is given by (bin sample count)/(total sample number)
* compute PSI value:
    psi = sum( (actual_percentage - expect_percentage) * ln(actual_percentage / expect_percentage) )
   
For more details of psi, you can refer to this [PSI tutorial](https://www.lexjansen.com/wuss/2017/47_Final_Paper_PDF.pdf)
    
#### Param

* max_bin_num: int, max bin number of quantile feature binning
* need_run: bool, need to run this module in DSL
* dense_missing_val: int/ float/ string
                     imputed missing value when input format is dense, default is set to np.nan. Default setting is 
                     suggested 

#### How to use

An example is offered in folder.