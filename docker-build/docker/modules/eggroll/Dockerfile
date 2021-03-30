ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

RUN set -eux; \
    rpm --rebuilddb; \
    rpm --import /etc/pki/rpm-gpg/RPM*; \
    yum install -y which strace java-1.8.0-openjdk java-1.8.0-openjdk-devel ; \
    yum clean all;

WORKDIR /data/projects/fate/eggroll/

ARG version

COPY eggroll ./
COPY python /data/projects/fate/python

ENV PYTHONPATH=/data/projects/fate/python:/data/projects/fate/eggroll/python
ENV EGGROLL_HOME=/data/projects/fate/eggroll/

CMD sh ./bin/eggroll.sh clustermanager start

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
