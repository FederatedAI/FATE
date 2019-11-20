FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

WORKDIR /data/projects/fate/meta-service
COPY eggroll-meta-service-1.1.tar.gz /data/projects/fate/meta-service/eggroll-meta-service-1.1.tar.gz

RUN tar -xzf eggroll-meta-service-1.1.tar.gz && \
    rm eggroll-meta-service-1.1.tar.gz && \
    ln -s eggroll-meta-service-1.1.jar fate-meta-service.jar
EXPOSE 8590
CMD java -cp "conf/:lib/*:fate-meta-service.jar" com.webank.ai.eggroll.framework.MetaService -c conf/meta-service.properties

