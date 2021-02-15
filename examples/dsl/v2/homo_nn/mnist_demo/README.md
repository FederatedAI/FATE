## How To

### use csv data

1. download and parse mnist data by

```bash
python ../../../../scripts/download_mnist_data.py
```

2. generate conf by

```bash
python build_mnist_testsuite.py
```

3. use fate_test to run generated testsuite

```
fate_test suite -i .
```

### use images directly

1. download and parse mnist data by

```bash
python ../../../../scripts/download_mnist_data_as_images.py
```

2. upload data

```bash
python ../../../../../python/fate_flow/fate_flow_client.py -f upload -c upload.json
```

3. submit job

```bash
python ../../../../../python/fate_flow/fate_flow_client.py -f submit_job -c mnist_conf.json -d mnist_dsl.json
```
