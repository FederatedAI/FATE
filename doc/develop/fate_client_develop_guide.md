# develop env

1.  create virtual env
    
    ``` sourceCode bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  install poetry
    
    ``` sourceCode bash
    pip install poetry
    ```

3.  install fate\_testsuite by poetry
    
    ``` sourceCode bash
    poetry install
    ```

4.  use flow client
    
    ``` sourceCode bash
    flow -h
    ```

# publish

1.  build
    
    ``` sourceCode bash
    poetry build
    ```

2.  use testpypi to avoid shipping broken versions of packages
    
    ``` sourceCode bash
    poetry config repositories.testpypi https://test.pypi.org/legacy/
    poetry publish -r testpypi
    ```

3.  viewing package on testpypi.pypi.org

4.  test version in a separate virtual
    environment:
    
    ``` sourceCode bash
    pip install --extra-index-url https://testpypi.python.org/pypi fate_testsuite
    ```

5.  publish to pypi
    
    ``` sourceCode bash
    poetry publish
    ```
