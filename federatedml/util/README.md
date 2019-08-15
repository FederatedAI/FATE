### DataIO

Data IO is the most basic component of Fate Algorithm module. 
It transforms the input DTable to a DTable whose values are Instance Object defined in federatedml.feature.instance, 
and what's more, the transformed Dtable is the input data format of all other algorithm module, such as intersect、
homo and hetero LR、SecureBoost and so on.

Data IO module accepts the following input data format and transforms them to desired output DTable.
* dense input format, input DTable's value item is a list of single element
(e.g. "1.0,2.0,3.0,4.5")
* svm-light input format, first item of input DTable's value is label, following by a list of complex "feature_id:value" items
    (e.g. value is "1 1:0.5 2:0.6")
* tag input format, the input DTable's value is a list of tag, data io module first aggregates all tags occurred in 
input table, then changes all input line to one-hot representation in sorting the occurred tags by lexicographic order
    (e.g. assume values is "a c", "a b d", after processing, the new values became "1 0 1 0", "1 1 0 1".)
* tag:value input format, the input DTable's value is a list of tag:value, like a mixed svm-light and tag input-format. 
data io module first aggregates all tags occurred in input table, then changes all input line to one-hot representation in 
sorting the occurred tags by lexicographic order, then fill the occur item with value.
    (e.g. assume values is "a:0.2 c:1.5", "a:0.3 b:0.6 d:0.7", after processing, the new values became "0.2 0 0.5 0", "0.3 0.6 0 0.7".)
    

#### Other Features of DataIO

* Missing value impute, provides ["mean", "designated", "min", "max"] methods to impute missing value
* Outlier value replace, also provides several outlier replace method like missing value impute.
Please check out federatedmd/feature/imputer.py for more details.



