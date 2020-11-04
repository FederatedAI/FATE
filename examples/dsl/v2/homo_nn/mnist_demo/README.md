## How To

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