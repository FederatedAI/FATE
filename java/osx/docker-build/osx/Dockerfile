FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine as builder

WORKDIR /data/projects/fate/osx/

RUN addgroup -S apps && adduser -S app -G apps

USER root

ARG version

COPY deploy/osx.tar.gz ./

RUN tar -xzvf osx.tar.gz && \
    rm osx.tar.gz

WORKDIR /data/projects/fate/osx/osx

CMD java -server  -XX:+UseG1GC -XX:G1HeapRegionSize=16m -XX:G1ReservePercent=25 -XX:InitiatingHeapOccupancyPercent=30 -XX:SoftRefLRUPolicyMSPerMB=0 -verbose:gc -Xloggc:/dev/shm/rmq_srv_gc_%p_%t.log -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintGCApplicationStoppedTime -XX:+PrintAdaptiveSizePolicy -XX:+UseGCLogFileRotation -XX:NumberOfGCLogFiles=5 -XX:GCLogFileSize=30m -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=./../oom/heapdump.hprof -XX:-OmitStackTraceInFastThrow -XX:+AlwaysPreTouch -XX:MaxDirectMemorySize=15g -XX:-UseLargePages -XX:-UseBiasedLocking -cp conf/broker/:lib/*:extension/*:./../osx-broker-1.0.0-alpha.jar org.fedai.osx.broker.Bootstrap -c /data/projects/fate/osx/osx/conf

EXPOSE 8767

