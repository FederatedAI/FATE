FROM gitpod/workspace-base:latest

### Java ###
USER gitpod
RUN curl -fsSL "https://get.sdkman.io" | bash \
 && bash -c ". /home/gitpod/.sdkman/bin/sdkman-init.sh \
             && sdk install java 8.0.292.j9-adpt \
             && sdk install maven \
             && sdk flush archives \
             && sdk flush temp \
             && mkdir /home/gitpod/.m2 \
             && printf '<settings>\n  <localRepository>/tmp/m2-repository/</localRepository>\n</settings>\n' > /home/gitpod/.m2/settings.xml \
             && echo 'export SDKMAN_DIR=\"/home/gitpod/.sdkman\"' >> /home/gitpod/.bashrc.d/99-java \
             && echo '[[ -s \"/home/gitpod/.sdkman/bin/sdkman-init.sh\" ]] && source \"/home/gitpod/.sdkman/bin/sdkman-init.sh\"' >> /home/gitpod/.bashrc.d/99-java \
             && sudo mkdir -p /fateboard && sudo chown gitpod:gitpod /fateboard"

### fateboard ###
USER gitpod
COPY --chown=gitpod:gitpod fateboard /fateboard/repo
RUN bash -c ". /home/gitpod/.sdkman/bin/sdkman-init.sh \
        && printf '<settings>\n  <localRepository>/tmp/m2-repository/</localRepository>\n</settings>\n' > /home/gitpod/.m2/settings.xml \
        && mvn -f /fateboard/repo/pom.xml package \
        && rm -rf /tmp/m2-repository \
        && printf '<settings>\n  <localRepository>/workspace/m2-repository/</localRepository>\n</settings>\n' > /home/gitpod/.m2/settings.xml" \
    && find /fateboard/repo/target -iname 'fateboard-*.jar' -exec cp {} /fateboard/fateboard.jar \; \
    && mkdir -p /fateboard/resources \
    && cp /fateboard/repo/src/main/resources/ssh.properties /fateboard/resources/ \
    && cp /fateboard/repo/src/main/resources/application.properties /fateboard/resources/ \
    && sed -i 's#fateboard.datasource.jdbc-url=.*#fateboard.datasource.jdbc-url=jdbc:sqlite:/workspace/FATE/fate_sqlite.db#' /fateboard/resources/application.properties \
    && sudo rm -rf /fateboard/repo

### Python ###
USER root
RUN apt-get update && apt-get install libgmp3-dev -y && apt-get install -y libmpfr-dev libmpfr-doc libmpfr6 && apt-get install libmpc-dev -y
RUN mkdir -p /venv && chown gitpod:gitpod /venv 

USER gitpod
RUN sudo install-packages python3-pip

COPY python/requirements.txt /venv/
ENV PIP_USER=
ENV PYTHONUSERBASE=
ENV PATH=$HOME/.pyenv/bin:$HOME/.pyenv/shims:$PATH
RUN curl -fsSL https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash \
    && { echo; \
        echo 'eval "$(pyenv init -)"'; \
        echo 'eval "$(pyenv virtualenv-init -)"'; } >> /home/gitpod/.bashrc.d/60-python \
    && pyenv update \
    && pyenv install 3.6.15 \
    && pyenv global 3.6.15 \
    && python3 -m venv /venv/py36 --system-site-packages \
    && /venv/py36/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /venv/py36/bin/python -m pip install --no-cache-dir --upgrade setuptools wheel virtualenv pipenv pylint rope flake8 \
        mypy autopep8 pep8 pylama pydocstyle bandit notebook twine jedi black isort \
    && /venv/py36/bin/python -m pip install --no-cache-dir -r /venv/requirements.txt \
    && sudo rm -rf /tmp/* \
    && sudo rm /venv/requirements.txt

COPY python/fate_client /venv/fate_client
RUN /venv/py36/bin/python -m pip install --no-cache-dir /venv/fate_client \
    && /venv/py36/bin/pipeline init --ip 127.0.0.1 --port 9380 \
    && /venv/py36/bin/flow init --ip 127.0.0.1 --port 9380 \
    && sudo rm -rf /tmp/*

COPY python/fate_test /venv/fate_test
RUN /venv/py36/bin/python -m pip install --no-cache-dir /venv/fate_test \
    && /venv/py36/bin/fate_test config \
    && sed -i 's#data_base_dir:.*#data_base_dir: /workspace/FATE#' /venv/py36/lib/python3.6/site-packages/fate_test/fate_test_config.yaml \
    && sed -i 's#fate_base:.*#fate_base: /workspace/FATE#' /venv/py36/lib/python3.6/site-packages/fate_test/fate_test_config.yaml \
    && sudo rm -rf /tmp/*