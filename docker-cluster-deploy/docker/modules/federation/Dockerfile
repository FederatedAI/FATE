ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

RUN mkdir -p /data/projects/fate
COPY . /data/projects/fate/federation
EXPOSE 9394

CMD cd /data/projects/fate/federation && \
    java -cp "conf/:lib/*:fate-federation.jar" com.webank.ai.fate.driver.Federation -c conf/federation.properties

