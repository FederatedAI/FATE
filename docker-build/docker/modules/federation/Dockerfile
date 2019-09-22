FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

RUN mkdir -p /data/projects/fate
COPY . /data/projects/fate/federation
EXPOSE 9394

CMD cd /data/projects/fate/federation && \
    java -cp "conf/:lib/*:fate-federation.jar" com.webank.ai.fate.driver.Federation -c conf/federation.properties

