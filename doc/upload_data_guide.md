## Upload Data Guide

Before start a modeling task, the data to be used should be uploaded. Typically, a party is usually a cluster which include multiple nodes. Thus, when we upload these data, the data will be allocated to those nodes.

### Accepted Data Type

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


### Define upload data config file

Here is an example showing how to create a upload config file:
```
{
  "file": "examples/data/breast_b.csv",
  "head": 1,
  "partition": 10,
  "work_mode": 0,
  "table_name": "hetero_breast_b",
  "namespace": "hetero_guest_breast"
}
```

Field Specifications:
1. file: file path
2. head: Specify whether your data file include a header or not
3. partition: Specify how many partitions used to store the data
4. local: Specify your current party info
5. table_name & namespace: Indicators for stored data table.

### Upload Command

We use fate-flow to upload data. The command is as follows:

> python ${your_install_path}/fate_flow/fate_flow_client.py -f upload -c dsl_test/upload_data.json

Note: This step is needed for every data-provide party(i.e. Guest and Host).

After running this command, the following information is shown if it is success.

```
{
    "data": {
        "namespace": "breast_hetero",
        "pid": 74684,
        "table_name": "breast_b"
    },
    "jobId": "20190801152750392991_436",
    "meta": null,
    "retcode": 0,
    "retmsg": "success",
    "created_at": "2019-08-01 15:27:50"
}
```

And as this output shown, the table_name and namespace has been list which can be taken as input config in submit-runtime conf.