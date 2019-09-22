ARG PREFIX=prefix
ARG BASE_TAG=base_tag
ARG BUILDER_TAG=builder_tag

# build image
FROM ${PREFIX}/storage-service-builder:${BUILDER_TAG}

FROM ${PREFIX}/base-image:${BASE_TAG}

# install which
RUN yum install -y which

# copy source
COPY storage-service /data/projects/fate/storage-service-cxx
COPY egg-service /data/projects/fate/egg
COPY egg-processor /data/projects/fate/python

COPY --from=0 /data/projects/fate/third_party/include /data/projects/fate/storage-service-cxx/third_party/include
COPY --from=0 /data/projects/fate/third_party/share /data/projects/fate/storage-service-cxx/third_party/share
COPY --from=0 /data/projects/fate/third_party/bin /data/projects/fate/storage-service-cxx/third_party/bin
COPY --from=0 /data/projects/fate/third_party/lib /data/projects/fate/storage-service-cxx/third_party/lib

# build storage-service
RUN cd /data/projects/fate/storage-service-cxx && \
    sed -ie "20s#-I. -I.*#-I. -I/data/projects/fate/storage-service-cxx/third_party/include#g" ./Makefile && \
    sed -ie "34s#LDFLAGS += -L.*#LDFLAGS += -L/data/projects/fate/storage-service-cxx/third_party/lib -llmdb -lboost_system -lboost_filesystem -lglog -lgpr#g" ./Makefile && \
    sed -ie "36s#PROTOC =.*#PROTOC = /data/projects/fate/storage-service-cxx/third_party/bin/protoc#g" ./Makefile && \
    sed -ie "37s#GRPC_CPP_PLUGIN =.*#GRPC_CPP_PLUGIN = /data/projects/fate/storage-service-cxx/third_party/bin/grpc_cpp_plugin#g" ./Makefile && \
    make

# create volume
VOLUME /data/projects/fate/data-dir

# work dir
WORKDIR /data/projects/fate

# start services
CMD cd /data/projects/fate/storage-service-cxx && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/data/projects/fate/storage-service-cxx/third_party/lib && \
    ./storage-service -p 7778 -d /data/projects/fate/data-dir >> logs/console.log 2>>logs/error.log & \
    cd /data/projects/fate/egg && \
    java -cp "conf/:lib/*:fate-egg.jar" com.webank.ai.fate.eggroll.Egg -c conf/egg.properties

