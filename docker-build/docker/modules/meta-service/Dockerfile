FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

WORKDIR /data/projects/fate/meta-service/

ARG version

COPY eggroll-meta-service-${version}.tar.gz ./
COPY conf ./conf

RUN tar -xzf eggroll-meta-service-${version}.tar.gz && \
    rm eggroll-meta-service-${version}.tar.gz && \
    ln -s eggroll-meta-service-*.jar fate-meta-service.jar
EXPOSE 8590

CMD java -cp "conf/:lib/*:fate-meta-service.jar" com.webank.ai.eggroll.framework.MetaService -c conf/meta-service.properties

