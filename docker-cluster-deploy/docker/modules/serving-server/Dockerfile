ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

WORKDIR /data/projects/fate/serving-server
COPY . /data/projects/fate/serving-server/
EXPOSE 6379
CMD java -cp "conf/:lib/*:fate-serving-server.jar" com.webank.ai.fate.serving.ServingServer -c conf/serving-server.properties

