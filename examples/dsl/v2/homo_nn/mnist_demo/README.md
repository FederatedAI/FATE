## How To

### use images directly

1. download and parse mnist data by

```bash
python ../../../../scripts/download_mnist_data_as_images.py
```

2. bind local path

```bash
a. replace $PROJECT_BASE in bind_local_path.json 
b. flow table bind -c bind_local_path.json
```

3. submit job

```bash
flow job submit -c mnist_conf.json -d mnist_dsl.json
```
