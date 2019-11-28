FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

RUN mkdir -p /data/projects/fate
COPY fate-federation-1.1.tar.gz /data/projects/fate/federation/fate-federation-1.1.tar.gz

RUN cd /data/projects/fate/federation/ && \
    tar -xzf fate-federation-1.1.tar.gz && \
    rm fate-federation-1.1.tar.gz && \
    ln -s fate-federation-1.1.jar fate-federation.jar
EXPOSE 9394

CMD cd /data/projects/fate/federation && \
    java -cp "conf/:lib/*:fate-federation.jar" com.webank.ai.fate.driver.Federation -c conf/federation.properties

