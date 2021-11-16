ARG PREFIX=prefix
ARG BASE_TAG=tag

FROM ${PREFIX}/base-image:${BASE_TAG} as builder
WORKDIR /data/projects/fate/
COPY eggroll.tar.gz .
COPY fate.tar.gz .
RUN tar -xzf eggroll.tar.gz; \
    tar -xzf fate.tar.gz;

RUN ls -l

FROM ${PREFIX}/base-image:${BASE_TAG}

RUN set -eux; \
    rpm --rebuilddb; \
    rpm --import /etc/pki/rpm-gpg/RPM*; \
    yum install -y which strace java-1.8.0-openjdk java-1.8.0-openjdk-devel ; \
    yum clean all;

WORKDIR /data/projects/fate/eggroll/

COPY --from=builder /data/projects/fate/eggroll /data/projects/fate/eggroll
COPY --from=builder /data/projects/fate/fate /data/projects/fate/fate

ENV PYTHONPATH=/data/projects/fate/fate/python:/data/projects/fate/eggroll/python
ENV EGGROLL_HOME=/data/projects/fate/eggroll/

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
