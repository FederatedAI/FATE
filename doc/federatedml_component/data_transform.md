# DataTransform

Data Transform is the most basic component of Fate Algorithm module. It
transforms the input Table to a Table whose values are Instance Object
defined [here](../../python/federatedml/feature/instance.py), and
what's more, the transformed table is the input data format of all other
algorithm module, such as intersect、 homo LR and hetero LR、SecureBoost
and so on.

Data IO module accepts the following input data format and transforms
them to desired output Table.

  - dense input format  
    input Table's value item is a list of single element, e.g. :
    
        1.0,2.0,3.0,4.5
        1.1,2.1,3.4,1.3
        2.4,6.3,1.5,9.0

  - svm-light input format  
    first item of input Table's value is label, following by a list of
    complex "feature\_id:value" items, e.g. :
    
        1 1:0.5 2:0.6
        0 1:0.7 3:0.8 5:0.2

  - tag input format  
    the input Table's value is a list of tag, data transform module first
    aggregates all tags occurred in input table, then changes all input
    line to one-hot representation in sorting the occurred tags by
    lexicographic order, e.g. assume values is :
    
        a c
        a b d
    
    after processing, the new values became: :
    
        1 0 1 0
        1 1 0 1

<!-- end list -->

  - :tag:value input format: the input Table's value is a list of
    <tag:value>, like a mixed svm-light and tag input-format. data transform
    module first aggregates all tags occurred in input table, then
    changes all input line to one-hot representation in sorting the
    occurred tags by lexicographic order, then fill the occur item with
    value. e.g. assume values is
    
        a:0.2 c:1.5
        a:0.3 b:0.6 d:0.7
    
    after processing, the new values became: :
    
        0.2 0 0.5 0
        0.3 0.6 0 0.7

<!-- mkdocs
## Param

::: federatedml.param.data_transform_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
 -->

## Other Features of DataTransform

  - Missing value impute, provides \["mean", "designated", "min",
    "max"\] methods to impute missing value
  - Outlier value replace, also provides several outlier replace method
    like missing value impute.
  - <font color="red">Parameters of data meta should be set when uploading or 
    binding data since FATE-v1.9.0, refer to upload guides please. </font>