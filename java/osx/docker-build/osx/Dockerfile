FROM eclipse/centos_jdk8 as builder

WORKDIR /data/projects/fate/osx/

# RUN addgroup -S apps && adduser -S app -G apps

USER root

ARG version

COPY deploy/osx.tar.gz ./

RUN tar -xzvf osx.tar.gz && \
    rm osx.tar.gz

# FROM mcr.microsoft.com/java/jre:8u192-zulu-centos
#
WORKDIR /data/projects/fate/osx/osx
CMD java -server   -cp /data/projects/fate/osx/osx/conf/broker/*:/data/projects/fate/osx/osx/lib/* org.fedai.osx.broker.Bootstrap -c /data/projects/fate/osx/osx/conf
EXPOSE 9377
EXPOSE 8089

