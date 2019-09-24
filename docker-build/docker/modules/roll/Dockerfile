FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

COPY . /data/projects/fate/roll

WORKDIR /data/projects/fate

CMD cd roll && \
    java -cp "conf/:lib/*:fate-roll.jar" com.webank.ai.fate.eggroll.Roll -c conf/roll.properties

