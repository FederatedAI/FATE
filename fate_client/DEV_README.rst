develop env
-----------

1. create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate


2. install poetry

   .. code-block:: bash

      pip install poetry


3. install fate_testsuite by poetry

   .. code-block:: bash

      poetry install


4. use flow client

   .. code-block:: bash

      flow -h


publish
--------

1. build

   .. code-block:: bash

      poetry build

2. use testpypi to avoid shipping broken versions of packages

   .. code-block:: bash

      poetry config repositories.testpypi https://test.pypi.org/legacy/
      poetry publish -r testpypi

3. viewing package on testpypi.pypi.org

4. test version in a separate virtual environment:

   .. code-block:: bash

      pip install --extra-index-url https://testpypi.python.org/pypi fate_testsuite


5. publish to pypi

   .. code-block:: bash

      poetry publish
