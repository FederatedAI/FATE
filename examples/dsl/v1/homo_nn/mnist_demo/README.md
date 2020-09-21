## How To

1. download and parse mnist data by
  ```bash
  python download_data.py
  ```

2. upload data to eggroll using config `upload_test.json` and `upload_train.json`

3. generate conf by
  ```bash
  python build_conf.py
  ```

4. submit job with conf `mnist_conf.json` and dsl `mnist_dsl.json`