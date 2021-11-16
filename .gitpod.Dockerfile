FROM gitpod/workspace-full:latest

### jdk
USER gitpod
RUN bash -c ". /home/gitpod/.sdkman/bin/sdkman-init.sh \
    && sed -i 's#sdkman_auto_answer=.*#sdkman_auto_answer=true#' /home/gitpod/.sdkman/etc/config \
    && sdk install java 8.0.292.j9-adpt \
    && sdk default java 8.0.292.j9-adpt"

USER root
RUN apt-get update && apt-get install libgmp3-dev -y && apt-get install -y libmpfr-dev libmpfr-doc libmpfr6 && apt-get install libmpc-dev -y
RUN mkdir -p /venv && chown gitpod:gitpod /venv \
    && mkdir -p /fateboard && chown gitpod:gitpod /fateboard

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

COPY python/fate_client /venv/fate_client
RUN /venv/py36/bin/python -m pip install --no-cache-dir /venv/fate_client \
    && /venv/py36/bin/pipeline init --ip 127.0.0.1 --port 9380

COPY python/fate_test /venv/fate_test
RUN /venv/py36/bin/python -m pip install --no-cache-dir /venv/fate_test