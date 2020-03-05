# Query Interface

### query schema

The related configuration is list at the beginning of query_schema.py file. If feature idx is -1, this interface will return all the header by combining idx and header name. Otherwise, if a list is provided, the listed header will be returned.

```
ROLE = 'host'
PARTY_ID = 10000
feature_idx = -1
# feature_idx = [0, 1, 2]
```

Example return result:
```
Queried header is [(0, '0'), (1, '2000000134'), (2, '2000024892'), (100, '2000027145')]
```

Start command:

> python query_schema.py -j {your_job_id}

This command will query output data result of component dataio_0, if you want to check out other component, -cpn parameter is required.