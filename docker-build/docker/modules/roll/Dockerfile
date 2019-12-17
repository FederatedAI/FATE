FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

WORKDIR /data/projects/fate/roll/

COPY eggroll-roll-*.tar.gz ./
COPY conf ./conf

RUN tar -xzf eggroll-roll-*.tar.gz && \
    rm eggroll-roll-*.tar.gz && \
    ln -s eggroll-roll-*.jar fate-roll.jar

CMD java -cp "conf/:lib/*:fate-roll.jar" com.webank.ai.eggroll.framework.Roll -c conf/roll.properties

