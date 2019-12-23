### SPDZ

SPDZ is a multiparty computation scheme based on somewhat homomorphic encryption (SHE). 

#### init
```python
from arch.api import session
from arch.api import federation
s = session.init("session_name", 0)
federation.init("session_name", {
    "local": {
        "role": "guest",
        "party_id": 1000
    },
    "role": {
        "host": [999],
        "guest": [1000]
    }
 })
partys = federation.all_parties()
# [Party(role=guest, party_id=1000), Party(role=host, party_id=999)]
```
#### spdz env
tensor should be created and processed in spdz env:
```python
from federatedml.secureprotol.spdz import SPDZ
with SPDZ() as spdz:
    ...
```
#### create tenser
We currently provide two implement of fixed point tensor:

1. one is based on numpy's array for non-distributed use:
    ```python
    from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
    ```
    - on guest side(assuming local Party is partys[0]): 
    ```python
        import numpy as np
        data = np.array([[1,2,3], [4,5,6]])
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", partys[1])
        ```

    - on host side(assuming PartyId is partys[1]):
    ```python
        import numpy as np
        data = np.array([[3,2,1], [6,5,4]])
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", partys[1])
    ```

2. one based on a table for distributed use:
    ```python
    from federatedml.secureprotol.spdz.tensor.fixedpoint_table import FixedPointTensor
    ```
    - on guest side(assuming PartyId is partys[0]): 
    ```python
        data = session.parallelize(np.array([1,2,3]), np.array([4,5,6]))
        with SPDZ() as spdz:
            x = FixedPointTensor.from_source("x", data)
            y = FixedPointTensor.from_source("y", party_1)
        ```

    - on host side(assuming PartyId is partys[1]):
    ```python
        data = session.parallelize(np.array([3,2,1]), np.array([6,5,4]))
        with SPDZ() as spdz:
            y = FixedPointTensor.from_source("y", data)
            x = FixedPointTensor.from_source("x", party_0)
    ```

When tensor created from a provided data, data is split into n shares and every party gets a different one. 
#### rescontruct
Value can be rescontructed from tensor

```python
x.get() # array([[1, 2, 3],[4, 5, 6]])
y.get() # array([[3, 2, 1],[6, 5, 4]])
```

#### add/miue
You can add or subtract tensors

```python
z = x + y
t = x - y
```
#### dot
You can do dot arithmetic:
```python
x.dot(y)
```

#### einsum (numpy version only)
When using numpy's tensor, powerful einsum arithmetic is available:
```python
x.einsum(y, "ij,kj->ik")  # dot
```

