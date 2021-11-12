FROM gitpod/workspace-full:latest


### Python ###
USER root
RUN apt-get update && apt-get install libgmp3-dev -y && apt-get install -y libmpfr-dev libmpfr-doc libmpfr6 && apt-get install libmpc-dev -y
RUN mkdir -p /venv && chown gitpod:gitpod /venv

USER gitpod
COPY python/requirements.txt /venv/
ENV PIP_USER=
ENV PYTHONUSERBASE=
RUN pyenv install 3.6.15 \
    && pyenv global 3.6.15 \
    && python3 -m venv /venv/py36 --system-site-packages \
    && /venv/py36/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /venv/py36/bin/python -m pip install --no-cache-dir --upgrade setuptools wheel virtualenv pipenv pylint rope flake8 \
        mypy autopep8 pep8 pylama pydocstyle bandit notebook twine jedi black isort \
    && /venv/py36/bin/python -m pip install --no-cache-dir -r /venv/requirements.txt \
    && sudo rm -rf /tmp/* \
    && sudo rm /venv/requirements.txt
RUN bash -c ". /home/gitpod/.sdkman/bin/sdkman-init.sh \
    && sed -i 's#sdkman_auto_answer=.*#sdkman_auto_answer=true#' /home/gitpod/.sdkman/etc/config \
    && sdk install java 8.0.292.j9-adpt \
    && sdk default java 8.0.292.j9-adpt"