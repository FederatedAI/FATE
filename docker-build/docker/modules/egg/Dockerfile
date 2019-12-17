ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

RUN set -eux; \
    rpm --rebuilddb; \
    rpm --import /etc/pki/rpm-gpg/RPM*; \
    yum install -y which java-1.8.0-openjdk java-1.8.0-openjdk-devel ; \
    yum clean all;

WORKDIR /data/projects/fate/eggroll/

COPY eggroll-computing-*.tar.gz ./
COPY eggroll-api-*.tar.gz ./
COPY eggroll-conf-*.tar.gz ./
COPY eggroll-egg-*.tar.gz ./
COPY eggroll-storage-service-cxx-*.tar.gz  ./
COPY third_party_eggrollv1.tar.gz ./
COPY federatedml ./python/federatedml
COPY arch ./python/arch
COPY fate_flow ./python/fate_flow
COPY conf ./egg/conf

RUN set -eux; \
    \
    cd /data/projects/fate/eggroll/; \
    mkdir -p egg python/eggroll/api python/eggroll/computing storage-service-cxx ; \
    tar -C ./egg -xvf eggroll-egg-*.tar.gz && rm eggroll-egg-*.tar.gz ; \
    tar -C ./python/eggroll/computing -xvf eggroll-computing-*.tar.gz && rm eggroll-computing-*.tar.gz ; \
    tar -C ./python/eggroll/api -xvf eggroll-api-*.tar.gz && rm eggroll-api-*.tar.gz ; \
    tar -C ./python/eggroll/conf -xvf eggroll-conf-*.tar.gz && rm eggroll-conf-*.tar.gz ; \
    tar -C ./storage-service-cxx -xvf eggroll-storage-service-cxx-*.tar.gz && rm eggroll-storage-service-cxx-*.tar.gz ; \
    tar -C ./storage-service-cxx -xvf third_party_eggrollv1.tar.gz && rm third_party_eggrollv1.tar.gz ; \
    cd storage-service-cxx ; \
    sed -i "20s#-I. -I.*#-I. -I/data/projects/fate/eggroll/storage-service-cxx/third_party/include#g" ./Makefile ; \
    sed -i "34s#LDFLAGS += -L.*#LDFLAGS += -L/data/projects/fate/eggroll/storage-service-cxx/third_party/lib -llmdb -lboost_system -lboost_filesystem -lglog -lgpr#g" ./Makefile ; \
    sed -i "36s#PROTOC =.*#PROTOC = /data/projects/fate/eggroll/storage-service-cxx/third_party/bin/protoc#g" ./Makefile ; \
    sed -i "37s#GRPC_CPP_PLUGIN =.*#GRPC_CPP_PLUGIN = /data/projects/fate/eggroll/storage-service-cxx/third_party/bin/grpc_cpp_plugin#g" ./Makefile ; \
    make; \
    cd /data/projects/fate/eggroll/egg && ln -s eggroll-egg-*.jar eggroll-egg.jar;

EXPOSE 7778 7888 

CMD cd /data/projects/fate/eggroll/storage-service-cxx && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/data/projects/fate/eggroll/storage-service-cxx/third_party/lib && \
    mkdir logs && \
    ./storage-service -p 7778 -d /data/projects/fate/data-dir >> logs/console.log 2>>logs/error.log & \
    cd /data/projects/fate/eggroll/egg && \
    java -cp "conf/:lib/*:eggroll-egg.jar" com.webank.ai.eggroll.framework.egg.Egg -c conf/egg.properties