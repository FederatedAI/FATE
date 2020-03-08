# session API Document


* 1. **[General API] (#1)**
* 2. **[Storage API] (#2)**
* 3. **[Computing API] (#3)**

---

## <a name="1">General API: session object</a>

### init
``` python
>>> init(job_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         backend: typing.Union[int, Backend] = Backend.EGGROLL)
```

Initializes session, should be called before all.

**Parameters:**

+ **job\_id** (string): Job id and default table namespace of this runtime.
+ **mode** (WorkMode): Set work mode,
    - standalone: `WorkMode.STANDALONE` or 0
    - cluster: `WorkMode.CLUSTER` or 1
+ **backend** (Backend): Set computing backend,
    - eggroll: `Backend.EGGROLL` or 0
    - spark: `Backend.SAPRK` or 1

**Returns:**

+ None

**Example:**


``` python
>>> from arch.api import session, WorkMode, Backend
>>> session.init("a_job_id", WorkMode.Standalone, Backend.EGGROLL)
```

### parallelize

``` python
parallelize(data, 
            include_key=False, 
            name=None,
            partition=1,
            namespace=None,
            perseistent=False, 
            create_if_missing=True, 
            error_if_exist=False, 
            chunk_size=100000, 
            in_place_computing=False)
```

Transforms an existing iterable data into a Table.

**Parameters:**

+ **data** (Iterable): Data to be put.
+ **include_key** (boolean): Whether to include key when parallelizing data into table.
+ **name** (string): Table name of result Table. A default table name will be generated when `None` is used
+ **partition** (int): Number of partitions when parallelizing data.
+ **namespace** (string): Table namespace of result Table. job_id will be used when `None` is used.
+ **create\_if\_missing** (boolean): Not implemented. Table will always be created.
+ **error\_if\_exist** (boolean): Not implemented. No error will be thrown if already exists.
+ **chunk_size** (int): Batch size when parallelizing data into Table.
+ **in\_place\_computing** (boolean): Whether in-place computing is enabled.

**Returns:**

+ **table** (Table): A Table consisting of parallelized data.

**Example:**

``` python
>>> from arch.api import session
>>> table = session.parallelize(range(10), in_place_computing=True)
```

### table

``` python
table(name, 
      namespace,
      partition=1, 
      create_if_missing=True, 
      error_if_exist=False,
      persistent=True,
      in_place_computing=False)
```

Loads an existing Table.

**Parameters:**

+ **name** (string): Table name of result Table.
+ **namespace** (string): Table namespace of result Table.
+ **partition** (int): Number of partitions when creating new Table.
+ **create\_if\_missing** (boolean): Not implemented. Table will always be created if not exists.
+ **error\_if\_exist** (boolean): Not implemented. No error will be thrown if already exists.
+ **persistent** (boolean): Where to load the Table, `True` from persistent storage and `False` from temporary storage.
+ **in\_place\_computing** (boolean): Whether in-place computing is enabled.

**Returns:**

+ **table** (Table): A Table consisting data loaded.

**Example:**

``` python
>>> from arch.api import session
>>> a = session.table('foo', 'bar', persistent=True)
```

### cleanup

``` python
cleanup(name, namespace, persistent=False)
```

Destroys Table(s). Wildcard can be used in `name` parameter.

**Parameters:**

+ **name** (string): Table name to be cleanup. Wildcard can be used here.
+ **namespace** (string): Table namespace to be cleanup. This needs to be a exact match.
+ **persistent** (boolean): Where to delete the Tables, `True` from persistent storage and `False` from temporary storage.

**Returns:**

+ None

**Example:**

``` python
>>> from arch.api import session
>>> session.cleanup('foo*', 'bar', persistent=True)
```

### generateUniqueId

```python
generateUniqueId()
```

Generates a unique ID each time it is invoked.

**Parameters:**

+ None

**Returns:**

+ **uniqueId** (string).

**Example:**

``` python
>>> from arch.api import session
>>> session.generateUniqueId()
```

### get_session_id

```python
get_session_id()
```

Returns session id.

**Parameters:**

+ None

**Returns:**

+ **session_id** (string).

**Example:**

``` python
>>> from arch.api import session
>>> session.get_session_id()
```

### get_data_table

```python
get_data_table(name, namespace)
```

Returns the table with given name and namespace.

**Parameters:**

+ **name(string)** table name
+ **namespace(string)** table namespace

**Returns:**

+ **table** (Table).

**Example:**

``` python
>>> from arch.api import session
>>> session.get_data_table(name, namespace)
```

### save_data
```python
save_data(kv_data: Iterable,
          name,
          namespace,
          partition=1,
          persistent: bool = True,
          create_if_missing=True,
          error_if_exist=False,
          in_version: bool = False,
          version_log=None)
```

Saves data to table, optional add version.

**Parameters:**

+ **kv_data(Iterable)** data to be saved 
+ **name(string)** table name
+ **namespace(string)** table namespace
+ **partition** (int): Number of partitions when creating new Table.
+ **create\_if\_missing** (boolean): Not implemented. Table will always be created if not exists.
+ **error\_if\_exist** (boolean): Not implemented. No error will be thrown if already exists.
+ **persistent** (boolean): Where to load the Table, `True` from persistent storage and `False` from temporary storage.
+ **in_version** (bool): add a version log or not
+ **version_log** (string): log to be added 
**Returns:**

+ **table** (Table).

**Example:**

``` python
>>> from arch.api import session
>>> session.save_data([("one", 1), ("two", 2)], "save_data", "readme", in_version=True, version_log="a version")
```

### save_data_table_meta:

```python
save_data_table_meta(kv, data_table_name, data_table_namespace):
```

Saves metas(in kv) to meta table associated with the table named `data_table_name` and namespaced `data_table_namespace`.

**Parameters:**

+ **kv(dict)** metas to save.
+ **data_table_name(string)** table name
+ **data_table_namespace(string)** table namespace

**Returns:**

+ None.

**Example:**

``` python
>>> from arch.api import session
>>> session.save_data_table_meta({"model_id": "a_id", "used_framework": "fate"}, "meta", "readme")
```

### get_data_table_meta:

```python
get_data_table_meta(key, data_table_name, data_table_namespace):
```

Gets meta keyed by `key` from meta table associated with table named `data_table_name` and namespaced `data_table_namespace`.

**Parameters:**

+ **key(string)** associated key.
+ **data_table_name(string)** table name
+ **data_table_namespace(string)** table namespace

**Returns:**

+ object associated with `key` provieded

**Example:**

``` python
>>> from arch.api import session
>>> session.get_data_table_meta("model_id", "meta", "readme") # a_id
```

### get_data_table_metas:

```python
get_data_table_meta(data_table_name, data_table_namespace):
```

Gets metas from meta table associated with table named `data_table_name` and namespaced `data_table_namespace`.

**Parameters:**

+ **data_table_name(string)** table name
+ **data_table_namespace(string)** table namespace

**Returns:**

+ metas(dict)

**Example:**

``` python
>>> from arch.api import session
>>> session.get_data_table_metas("meta", "readme") # {'model_id': 'a_id', 'used_framework': 'fate'}
```

### stop

```python
stop()
```

Stops session, clean all tables associated with this session.

**Parameters:**

+ None

**Returns:**

+ **session_id** (string).

**Example:**

```python
>>> from arch.api import session
>>> session.stop() 
```

## <a name="2">Storage API: class Table</a>

### collect
```python
collect(min_chunk_size=0)
``` 

Returns an iterator of (key, value) 2-tuple from the Table.

**Parameters:**

+ **min\_chunk\_size**: Minimum chunk size (key bytes + value bytes) returned if end of table is not hit. 0 indicates a default chunk size (partition_num * 1.75 MB); negative number indicates no chunk limit, i.e. returning all records. Default chunk size is recommended if there is no special needs from user.

**Returns:**

+ **iter** (Iterator)

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> b = a.collect(min_chunk_size=1000)
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
```

### count

```python
count()
```

Returns the number of elements in the Table.

**Parameters:**

+ None

**Returns:**

+ **num** (int): Number of elements in this Table.

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> a.count()
10
```

### delete

```python
delete(k)
```

Returns the deleted value corresponding to the key.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.

**Returns:**

+ **v** (object): Corresponding value of the deleted key. Returns None if key does not exist.

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> a.delete(1)
1
```

### destroy

```python
destroy()
```

Destroys this Table, freeing its associated storage resources.

**Parameters:**

+ None

**Returns:**

+ None

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> a.destroy()
```

### first

```python
first(keysOnly=False)
```

Returns the first element of a Table. Shortcut of `take(1, keysOnly)`

**Parameters:**

+ **keysOnly** (boolean): Whether to return keys only. `True` returns keys only and `False` returns both keys and values.

**Returns:**

+ **first_element** (tuple / object): First element of the Table. It is a tuple if `keysOnly=False`, or an object if `keysOnly=True`.

**Example:**

``` python
>>> a = session.parallelize([1, 2, 3])
>>> a.first()
(1, 1)
```

### get

```python
get(k)
```

Fetches the value matching key.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.

**Returns:**

+ **v** (object): Corresponding value of the key. Returns None if key does not exist.

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> a.get(1)
(1, 1)
```

### put

```python
put(k, v)
```

Stores a key-value record.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.
+ **v** (object): Value object. Will be serialized. Must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> a.put('hello', 'world')
>>> b = a.collect()
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), ('hello', 'world')]
```

### put_all

```python
put_all(kv)
```

Puts (key, value) 2-tuple stream from the iterable items. Elements must be exact 2-tuples, they may not be of any other type, or tuple subclass.

**Parameters:**

+ **kv** (Iterable): Key-Value 2-tuple iterable. Will be serialized. Each key must be less than 512 bytes, value must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = session.table('foo', 'bar')
>>> t = [(1, 2), (3, 4), (5, 6)]
>>> a.put_all(t)
>>> list(a.collect())
[(1, 2), (3, 4), (5, 6)]
```

### put_if_absent

>`put_if_absent(k, v)`

Stores a key-value record only if the key is not set.

**Parameters:**

+ **k** (object): Key object. Will be serialized. Must be less than 512 bytes.
+ **v** (object): Value object. Will be serialized. Must be less than 32 MB.

**Returns:**

+ None

**Example:**

``` python
>>> a = sessiojn.parallelize(range(10))
>>> a.put_if_absent(1, 2)
>>> b = a.collect()
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
>>> a.put_if_absent(-1, -1)
>>> list(b)
[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (-1, -1)]
```

### save_as

```python
save_as(name, namespace, partition=1)
```

Transforms a temporary table to a persistent table.

**Parameters:**

+ **name** (string): Table name of result Table.
+ **namespace** (string): Table namespace of result Table.
+ **partition** (int): Number of partition for the new persistent table.

**Returns:**

+ **table** (Table): Result persistent Table.

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> b = a.save_as('foo', 'bar', partition=2)
```

### take

```python
take(n, keysOnly=False)
```

Returns the first n element(s) of a Table. 

**Parameters:**

+ **n** (int): Number of top data returned.
+ **keysOnly** (boolean): Whether to return keys only. `True` returns keys only and `False` returns both keys and values.

**Returns:**

+ **result\_list** (list): Lists of top n keys or key-value pairs.

**Example:**

``` python
>>> a = session.parallelize([1, 2, 3])
>>> a.take(2)
[(1, 1), (2, 2)]
>>> a.take(2, keysOnly=True)
[1, 2]
```

## <a name="3">Computing API: class Table</a>

### filter
```python
filter(func)
```

Returns a new Table containing only those keys which satisfy a predicate passed in via `func`.

In-place computing does not apply.

**Parameters:**

+ **func** (k, v -> boolean): Predicate function returning a boolean.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> a = session.parallelize([0, 1, 2])
>>> b = a.filter(lambda k, v : k % 2 == 0)
>>> list(b.collect())
[(0, 0), (2, 2)]
>>> c = a.filter(lambda k, v : v % 2 != 0)
>>> list(c.collect())
[(1, 1)]
```

### flatMap

```python
flatMap(func)
```

Returns a new Table by first applying func, then flattening it.

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
>>> a = session.parallelize(range(5))
>>> b = a.flatMap(foo)
>>> list(b.collect())
[(83030, 83030), (84321, 84321), (84322, 84322), (91266, 91266), (91267, 91267), (91268, 91268), (91269, 91269), (71349, 71349), (71350, 71350), (71351, 71351)]
```

### glom

```python
glom()
```

Coalesces all elements within each partition into a list.

**Parameters:**

+ None

**Returns:**

+ **result\_list** (list): A list containing all coalesced partition and its elements. First element of each tuple is chosen from key of last element of each partition.

**Example:**

```
>>> a = session.parallelize(range(5), partition=3).glom().collect()
>>> list(a)
[(2, [(2, 2)]), (3, [(0, 0), (3, 3)]), (4, [(1, 1), (4, 4)])]
```

### join

```python
join(other, func)
```

Returns a Table containing all pairs of elements with matching keys in self and other, i.e. 'inner join'.

Each pair of elements will be returned as a (k, func(v1, v2)) tuple, where (k, v1) is in self and (k, v2) in other.

In-place computing applies if enabled. Results will be in left Table (the caller).

**Parameters:**

+ **other** (Table): Another Table to be operated with.
+ **func** (v1, v2 -> v): Binary operator applying to values whose key exists in both Tables.

**Returns:**

+ **table** (Table): A Table containing results.

**Example:**


``` python
>>> a = session.parallelize([('a', 1), ('b', 4)], include_key=True)
>>> b = session.parallelize([('a', 2), ('c', 3)], include_key=True)
>>> c = a.join(b, lambda v1, v2: v1 + v2)
>>> list(c.collect())
[('a', 3)]

```

### map

```python
map(func)
```

Returns a new Table by applying a function to each (key, value) 2-tuple of this Table.

In-place computing does not apply.

**Parameters:**

+ **func** (k1, v1 -> k2, v2): The function applying to each 2-tuple.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> a = session.parallelize(['a', 'b', 'c'])    # [(0, 'a'), (1, 'b'), (2, 'c')]
>>> b = a.map(lambda k, v: (v, v + '1'))
>>> list(b.collect())
[("a", "a1"), ("b", "b1"), ("c", "c1")]
```

### mapPartitions

```python
mapPartitions(func)
```

Returns a new Table by applying a function to each partition of this Table.

In-place computing does not apply.

**Parameters:**

+ **func** (iter -> v): The function applying to each partition.

**Returns:**

+ **table** (Table): A new table with k-v: uuid key - v.

**Example:**

``` python
>>> a = session.parallelize([1, 2, 3, 4, 5], partition=2)
>>> def f(iterator):
>>> 	sum = 0
>>> 	for k, v in iterator:
>>> 		sum += v
>>> 	return sum
>>> b = a.mapPartitions(f)
>>> list(b.collect())
[(3, 6), (4, 9)]
```

### 


```python
mapPartitions2(func)
```

Returns a new Table by applying a function to each partition of this Table.

In-place computing does not apply.

**Parameters:**

+ **func** (iter -> (k, v)): The function applying to each partition.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> a = session.parallelize([1, 2, 3, 4, 5], partition=2)
>>> def f(iterator):
>>> 	s = 0
>>> 	for k, v in iterator:
>>> 		s += v
>>> 	return [(s, s)]
>>> b = a.mapPartitions2(f)
>>> list(b.collect())
[(6, 6), (9, 9)]
```

### mapValue

```python
mapValues(func)
```

Returns a Table by applying a function to each value of this Table, while keys do not change.

In-place computing applies if enabled.

``` python
>>> a = session.parallelize([('a', ['apple', 'banana', 'lemon']), ('b', ['grapes'])], include_key=True)
>>> b = a.mapValues(lambda x: len(x))
>>> list(b.collect())
[('a', 3), ('b', 1)]
```

**Parameters:**

+ **func** (v1 -> v2): The function applying to each value.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> a = session.parallelize(['a', 'b', 'c'])    # [(0, 'a'), (1, 'b'), (2, 'c')]
>>> b = a.mapValues(lambda v: v + '1')
>>> list(b.collect())
[(0, 'a1'), (1, 'b1'), (2, 'c1')]
```

### reduce

```python
reduce(func, key_func=None)
```

Reduces the elements of this Table using the specified associative binary operator. Currently reduces partitions locally.

In-place computing does not apply.

**Parameters:**

+ **func** (v1, v2 -> v): Binary operator applying to each 2-tuple.
+ **key_func** (k -> k'): Unary operator applying to each key to obtain the real key for reducing. Defaults to None, which means reducing on original key.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> from operator import add
>>> session.parallelize([1, 2, 3, 4, 5]).reduce(add)
>>> 15

```

### sample

```python
sample(fraction, seed)
```

Return a sampled subset of this Table. 

In-place computing does not apply.

**Parameters:**
 
+ **fraction** (float): Expected size of the sample as a fraction of this Table's size without replacement: probability that each element is chosen. Fraction must be [0, 1] with replacement: expected number of times each element is chosen.
+ **seed** (float): Seed of the random number generator. Use current timestamp when `None` is passed.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> x = session.parallelize(range(100), partition=4)
>>>  6 <= x.sample(0.1, 81).count() <= 14
True
```

### subtractByKey

```python
subtractByKey(other : Table)
```

Returns a new Table containing elements only in this Table but not in the other Table. 

In-place computing applies if enabled. Result will be in left Table (the caller).

**Parameters:**

+ **other** (Table): Another Table to be operated with.

**Returns:**

+ **table** (Table): A new table containing results.

**Example:**

``` python
>>> a = session.parallelize(range(10))
>>> b = session.parallelize(range(5))
>>> c = a.subtractByKey(b)
>>> list(c.collect())
[(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
```

### union

```python
union(other, func=lambda v1, v2 : v1)
```

Returns union of this Table and the other Table. Function will be applied to values of keys that exist in both table.

In-place computing applies if enabled. Result will be in left Table (the caller).

**Parameters:**

+ **other** (Table): Another Table to be operated with.
+ **func** (v1, v2 -> v): The function applying to values whose key exists in both Tables. Default using left table's value.

**Returns:**

+ **table** (Table): A table containing results.

**Example:**

``` python
>>> a = session.parallelize([1, 2, 3])	# [(0, 1), (1, 2), (2, 3)]
>>> b = session.parallelize([(1, 1), (2, 2), (3, 3)])
>>> c = a.union(b, lambda v1, v2 : v1 + v2)
>>> list(c.collect())
[(0, 1), (1, 3), (2, 5), (3, 3)]
```

