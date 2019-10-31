# Federation API Document


## <a name="4">Federation API: cluster federation object</a>


>`init(job_id, runtime_conf)`

Initializes federation module. This method should be called before calling other federation APIs.

**Parameters:**

+ **job\_id** (string): Job id and default table namespace of this runtime.
+ **runtime\_conf** (dict):
 1. key 'local' maps to the role and party_id of current process.
 2. key 'role' maps to a dict mapping from each role to all involving party_ids.
 
 example runtime\_conf:


 ```
   {
      "local": {
        "role": "host",
        "party_id": 1000
      },
      "role": {
        "host": [999, 1000, 1001],
        "guest": [1002]
      }
   }
 ```

**Returns:**

+ None

**Example:**

``` python
>>> from arch.api import federation
>>> federation.init('job_id', runtime_conf)
```

--

>`get(name, tag, idx=-1)`

Gets data from other participant. This method will block until the remote object is fetched.

**Parameters:**

+ **name** (string): {alogrithm}.{variableName} defined in `transfer_conf.json`.
+ **tag** (string): Object version. Version often indicates epoch number in algorithms.
+ **idx** (int): Index of the party_ids in runtime role list. If out-of-range index is specified, list of all objects will be returned.

**Returns:**

+ **o** (object): The object itself if idx is in range, else return list of all objects from all possible sources.

**Example:**

``` python
>>> b = federation.get("RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag), idx=-1)
```

--

>`remote(obj, name, tag, role=None, idx=-1)`

Sends data to other parties. This method does not block.

**Parameters:**

+ **obj** (object): The object to send. This obj will be pickled.
+ **name** (string): {alogrithm}.{variableName} defined in `transfer_conf.json`.
+ **tag** (string): Object version. Version often indicates epoch number in algorithms.
+ **role** (string): The role you want to send data to.
+ **idx** (string): The idx of the party_ids of the role, if out-of-range, will send to all parties of the role.

**Returns:**

+ None

**Example:**

``` python
>>> federation.remote(a, "RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag))
```
