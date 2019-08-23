ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/meta-service
COPY . /data/projects/fate/meta-service
EXPOSE 8590
CMD java -cp "conf/:lib/*:fate-meta-service.jar" com.webank.ai.fate.eggroll.MetaService -c conf/meta-service.properties

