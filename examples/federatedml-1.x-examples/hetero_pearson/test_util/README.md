## How To

on guest side:
```bash
python make_fake_data_testsuite.py guest aname row1 col1 row2 col2 ...
```

and one host side

```bash
python make_fake_data_testsuite.py host aname row1 col1 row2 col2 ...
```

this create fake data with shapes:

```
(row1, col1)
(row2, col2)
...
```

and corresponding configures and a testsuite.json for testsuite.