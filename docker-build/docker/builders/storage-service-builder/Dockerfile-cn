ARG PREFIX=prefix
ARG BASE_TAG=base_tag

FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/

# update third party
RUN set -eux; \
    curl https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/third_party.tar.gz > third_party.tar.gz; \
    mkdir /data/projects/fate/third_party; \
    tar -C /data/projects/fate/third_party -xzvf third_party.tar.gz; \
    rm -f third_party.tar.gz
