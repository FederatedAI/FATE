ARG PREFIX=prefix
ARG BASE_TAG=tag
FROM ${PREFIX}/base-image:${BASE_TAG}

COPY . /data/projects/fate/roll

WORKDIR /data/projects/fate

CMD cd roll && \
    java -cp "conf/:lib/*:fate-roll.jar" com.webank.ai.fate.eggroll.Roll -c conf/roll.properties

