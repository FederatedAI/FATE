# EggRoll API Document


* 1. **[General API] (#1)**
* 2. **[Storage API] (#2)**
* 3. **[Computing API] (#3)**

---

## <a name="1">General API: eggroll object</a>

>``` python
>init(job_id=None, mode=WorkMode.STANDALONE, naming_policy=NamingPolicy.DEFAULT)
>```

Initializes EggRoll runtime. This API should always be called before other API calls.

**Parameters:**

+ **job\_id** (string): Job id and default table namespace of this runtime.
+ **mode** (WorkMode): Set runtime naming policy, `WorkMode.STANDALONE` or `WorkMode.CLUSTER`.
+ **naming\_policy** (NamingPolicy): Set runtime naming policy, `NamingPolicy.DEFAULT` or `NamingPolicy.ITER_AWARE`.

**Returns:**

+ None

**Example:**

``` python
>>> from arch import eggroll
>>> from eggroll.api import WorkMode
>>> from eggroll.api import NamingPolicy
>>> eggroll.init('a', WorkMode.CLUSTER, NamingPolicy.ITER_AWARE)
```

--

>``` python
>parallelize(data, include_key=False, name=None, partition=1,              
>             namespace=None, perseistent=False, create_if_missing=True, 
>             error_if_exist=False, chunk_size=100000, in_place_computing=False)
> ```

Takes an existing iterable dataset and transform it into a DTable.

**Parameters:**

+ **data** (Iterable): Data to be put into DTable.
+ **include_key** (boolean): Whether to include key when parallelizing data into dtable.
+ **name** (string): Table name of result DTable. A default table name will be generated when `None` is used
+ **partition** (int): Number of partitions when parallelizing data.
+ **namespace** (string): Table namespace of result DTable. job_id will be used when `None` is used.
+ **create\_if\_missing** (boolean): Not implemented. DTable will always be created.
+ **error\_if\_exist** (boolean): Not implemented. No error will be thrown if already exists.
+ **chunk_size** (int): Batch size when parallelizing data into DTable.
+ **in\_place\_computing** (boolean): Whether in-place computing is enabled.

**Returns:**

+ **dtable** (DTable): A DTable consisting data parallelized.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10), in_place_computing=True)
```

--

>``` python
>table(name, namespace, partition=1, create_if_missing=True, 
>       error_if_exist=False, persistent=True, in_place_computing=False)
>```

Loads an existing DTable.

**Parameters:**

+ **name** (string): Table name of result DTable.
+ **namespace** (string): Table namespace of result DTable.
+ **partition** (int): Number of partitions when creating new DTable.
+ **create\_if\_missing** (boolean): Not implemented. DTable will always be created if not exists.
+ **error\_if\_exist** (boolean): Not implemented. No error will be thrown if already exists.
+ **persistent** (boolean): Where to load the DTable, `True` from persistent storage and `False` from temporary storage.
+ **in\_place\_computing** (boolean): Whether in-place computing is enabled.

**Returns:**

+ **dtable** (DTable): A DTable consisting data loaded.

**Example:**

``` python
>>> a = eggroll.table('foo', 'bar', persistent=True)
```

--

>``` python
>cleanup(name, namespace, persistent=False)
>```

Destroys DTable(s). Wildcard can be used in `name` parameter.

**Parameters:**

+ **name** (string): Table name to be cleanup. Wildcard can be used here.
+ **namespace** (string): Table namespace to be cleanup. This needs to be a exact match.
+ **persistent** (boolean): Where to delete the DTables, `True` from persistent storage and `False` from temporary storage.

**Returns:**

+ None

**Example:**

``` python
>>> eggroll.cleanup('foo*', 'bar', persistent=True)
```

--

>`generateUniqueId()`

Generates a unique ID each time it is invoked.

**Parameters:**

+ None

**Returns:**

+ **uniqueId** (string).

**Example:**

``` python
>>> eggroll.generateUniqueId()
'_EggRoll_a_VM_centos_1560843036.15374231338500976562_66698'
```

--

## <a name="2">Storage API: class DTable</a>

>`collect(min_chunk_size=0)` 

Returns an iterator of (key, value) 2-tuple from the DTable. 

**Parameters:**

+ **min\_chunk\_size**: Minimum chunk size (key bytes + value bytes) returned if end of table is not hit. 0 indicates a default chunk size (partition_num * 1.75 MB); negative number indicates no chunk limit, i.e. returning all records. Default chunk size is recommended if there is no special needs from user.

**Returns:**

+ **iter** (_EggRollIterator)

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> b = a.collect(min_chunk_size=1000)
>>> print(b)
<eggroll.api.cluster.eggroll._EggRollIterator object at 0x7f29d5fa1fd0>
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
```

--

>`count()`

Returns the number of elements in the DTable.

**Parameters:**

+ None

**Returns:**

+ **num** (int): Number of elements in this DTable.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.count()
10
```

--

>`delete(k)`

Returns the deleted value corresponding to the key.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.

**Returns:**

+ **v** (object): Corresponding value of the deleted key. Returns None if key does not exist.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.delete(1)
1
```

--

>`destroy()`

Destroys this DTable, freeing its associated storage resources.

**Parameters:**

+ None

**Returns:**

+ None

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.destroy()
```

--

>`first(keysOnly=False)`

Returning the first element of a DTable. Shortcut of `take(1, keysOnly)`

**Parameters:**

+ **keysOnly** (boolean): Whether to return keys only. `True` returns keys only and `False` returns both keys and values.

**Returns:**

+ **first_element** (tuple / object): First element of the DTable. It is a tuple if `keysOnly=False`, or an object if `keysOnly=True`.

**Example:**

``` python
>>> a = eggroll.parallelize([1, 2, 3])
>>> a.first()
(1, 1)
```

--

>`get(k)`

Fetches the value matching key.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.

**Returns:**

+ **v** (object): Corresponding value of the key. Returns None if key does not exist.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.get(1)
(1, 1)
```

--

>`put(k, v)`

Stores a key-value record.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.
+ **v** (object): Value object. Will be serialized. Must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.put('hello', 'world')
>>> b = a.collect()
>>> print(b)
<eggroll.api.cluster.eggroll._EggRollIterator object at 0x7f29d5fa1fd0>
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), ('hello', 'world')]
```

--

>`put_all(kv)`

Puts (key, value) 2-tuple stream from the iterable items. Elements must be exact 2-tuples, they may not be of any other type, or tuple subclass.

**Parameters:**

+ **kv** (Iterable): Key-Value 2-tuple iterable. Will be serialized. Each key must be less than 512 bytes, value must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = eggroll.table('foo', 'bar')
>>> t = [(1, 2), (3, 4), (5, 6)]
>>> a.put_all(t)
>>> list(a.collect())
[(1, 2), (3, 4), (5, 6)]
```

--

>`put_if_absent(k, v)`

Stores a key-value record only if the key is not set.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.
+ **v** (object): Value object. Will be serialized. Must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> a.put(1, 1)
>>> b = a.collect()
>>> print(b)
<eggroll.api.cluster.eggroll._EggRollIterator object at 0x7f29d5fa1fd0>
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
```

--

>`save_as(name, namespace, partition=1)`

Transforms a temporary table to a persistent table.

**Parameters:**

+ **name** (string): Table name of result DTable.
+ **namespace** (string): Table namespace of result DTable.
+ **partition** (int): Number of partition for the new persistent table.

**Returns:**

+ **dtable** (DTable): Result persistent DTable.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> b = a.save_as('foo', 'bar', partition=2)
>>> print(b)
storage_type: LMDB, namespace: bar, name: foo, partitions: 2, in_place_computing: False
```

--

>`take(n, keysOnly=False)`

Returns the first n element(s) of a DTable. 

**Parameters:**

+ **n** (int): Number of top data returned.
+ **keysOnly** (boolean): Whether to return keys only. `True` returns keys only and `False` returns both keys and values.

**Returns:**

+ **result\_list** (list): Lists of top n keys or key-value pairs.

**Example:**

``` python
>>> a = eggroll.parallelize([1, 2, 3])
>>> a.take(2)
[(1, 1), (2, 2)]
>>> a.take(2, keysOnly=True)
[1, 2]
```

---

## <a name="3">Computing API: class DTable</a>

>`filter(func)`

Returns a new DTable containing only those keys which satisfy a predicate passed in via `func`.

In-place computing does not apply.

**Parameters:**

+ **func** (k, v -> boolean): Predicate function returning a boolean.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize([0, 1, 2])
>>> b = a.filter(lambda k, v : k % 2 == 0)
>>> list(b.collect())
[(0, 0), (2, 2)]
>>> c = a.filter(lambda k, v : v % 2 != 0)
>>> list(c.collect())
[(1, 1)]
```

--

>`flatMap(func)`

Returns a new DTable by first applying func, then flattening it.

In-place computing does not apply.

**Parameters:**

+ **func** (k, v -> list): The function applying to each 2-tuple.

**Returns:**

+ **result\_list** (list): A list containing all flattened elements within each list after applying `func`. 'Last-Write Win' policy is used if key exists in multiple lists.

**Example:**

``` python
>>> import random
>>> def foo(k, v):
...     result = []
...     r = random.randint(10000, 99999)
...     for i in range(0, k):
...         result.append((k + r + i, v + r + i))
...     return result
>>> a = eggroll.parallelize(range(5))
>>> b = a.flatMap(foo)
>>> list(b.collect())
[(83030, 83030), (84321, 84321), (84322, 84322), (91266, 91266), (91267, 91267), (91268, 91268), (91269, 91269), (71349, 71349), (71350, 71350), (71351, 71351)]
```

--

>`glom()`

Coalescing all elements within each partition into a list.

**Parameters:**

+ None

**Returns:**

+ **result\_list** (list): A list containing all coalesced partition and its elements. First element of each tuple is chosen from key of last element of each partition.

**Example:**

```
>>> a = eggroll.parallelize(range(5), partition=3).glom().collect()
>>> list(a)
[(2, [(2, 2)]), (3, [(0, 0), (3, 3)]), (4, [(1, 1), (4, 4)])]
```

--

>`join(other, func)`

Return an DTable containing all pairs of elements with matching keys in self and other, i.e. 'inner join'.

Each pair of elements will be returned as a (k, func(v1, v2)) tuple, where (k, v1) is in self and (k, v2) is in other.

In-place computing applies if enabled. Result will be in left DTable (the caller).

**Parameters:**

+ **other** (DTable): Another DTable to be operated with.
+ **func** (v1, v2 -> v): Binary operator applying to values whose key exists in both DTables.

**Returns:**

+ **dtable** (DTable): A DTable containing results.

**Example:**


``` python
>>> a = eggroll.parallelize([('a', 1), ('b', 4)], include_key=True)
>>> b = eggroll.parallelize([('a', 2), ('c', 3)], include_key=True)
>>> c = a.join(b, lambda v1, v2: v1 + v2)
>>> list(c.collect())
[('a', 3)]

```

--

>`map(func)`

Return a new DTable by applying a function to each (key, value) 2-tuple of this DTable.

In-place computing does not apply.

**Parameters:**

+ **func** (k1, v1 -> k2, v2): The function applying to each 2-tuple.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize(['a', 'b', 'c'])    # [(0, 'a'), (1, 'b'), (2, 'c')]
>>> b = a.map(lambda k, v: (v, v + '1'))
>>> list(b.collect())
[("a", "a1"), ("b", "b1"), ("c", "c1")]
```
--

>`mapPartitions(func)`

Return a new DTable by applying a function to each partition of this DTable.

In-place computing does not apply.

**Parameters:**

+ **func** ((k1, v1), (k2, v2) -> (k, v)): The function applying to each partition.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize([1, 2, 3, 4, 5], partition=2)
>>> def f(iterator):
>>> 	sum = 0
>>> 	for k, v in iterator:
>>> 		sum += v
>>> 	return sum
>>> b = a.mapPartitions(f)
>>> list(b.collect())
[(3, 6), (4, 9)]
```

--

>`mapValues(func)`

Return a DTable by applying a function to each value of this DTable, while keys does not change.

In-place computing applies if enabled.

``` python
>>> a = eggroll.parallelize([('a', ['apple', 'banana', 'lemon']), ('b', ['grapes'])], include_key=True)
>>> b = a.mapValues(lambda x: len(x))
>>> list(b.collect())
[('a', 3), ('b', 1)]
```

**Parameters:**

+ **func** (v1 -> v2): The function applying to each value.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize(['a', 'b', 'c'])    # [(0, 'a'), (1, 'b'), (2, 'c')]
>>> b = a.mapValues(lambda v: v + '1')
>>> list(b.collect())
[(0, 'a1'), (1, 'b1'), (2, 'c1')]
```

--

>`reduce(func)`

Reduces the elements of this DTable using the specified associative binary operator. Currently reduces partitions locally.

In-place computing does not apply.

**Parameters:**

+ **func** (v1, v2 -> v): Binary operator applying to each 2-tuple.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> from operator import add
>>> eggroll.parallelize([1, 2, 3, 4, 5]).reduce(add)
>>> 15

```

--

>`sample(fraction, seed)`

Return a sampled subset of this DTable. 

In-place computing does not apply.

**Parameters:**
 
+ **fraction** (float): Expected size of the sample as a fraction of this DTable's size without replacement: probability that each element is chosen. Fraction must be [0, 1] with replacement: expected number of times each element is chosen.
+ **seed** (float): Seed of the random number generator. Use current timestamp when `None` is passed.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> x = eggroll.parallelize(range(100), partition=4)
>>>  6 <= x.sample(0.1, 81).count() <= 14
True
```

--

>`subtractByKey(other : DTable)`

Returns a new DTable containing elements only in this DTable but not in the other DTable. 

In-place computing applies if enabled. Result will be in left DTable (the caller).

**Parameters:**

+ **other** (DTable): Another DTable to be operated with.

**Returns:**

+ **dtable** (DTable): A new table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize(range(10))
>>> b = eggroll.parallelize(range(5))
>>> c = a.subtractByKey(b)
>>> list(c.collect())
[(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
```

--

>`union(other, func=lambda v1, v2 : v1)`

Returns union of this DTable and the other DTable. Function will be applied to values of keys exist in both table.

In-place computing applies if enabled. Result will be in left DTable (the caller).

**Parameters:**

+ **other** (DTable): Another DTable to be operated with.
+ **func** (v1, v2 -> v): The function applying to values whose key exists in both DTables. Default using left table's value.

**Returns:**

+ **dtable** (DTable): A table containing results.

**Example:**

``` python
>>> a = eggroll.parallelize([1, 2, 3])	# [(0, 1), (1, 2), (2, 3)]
>>> b = eggroll.parallelize([(1, 1), (2, 2), (3, 3)])
>>> c = a.union(b, lambda v1, v2 : v1 + v2)
>>> list(c.collect())
[(0, 1), (1, 3), (2, 5), (3, 3)]
```

---

