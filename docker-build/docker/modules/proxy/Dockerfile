FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

RUN mkdir -p /data/projects/fate

COPY fate-proxy-1.1.tar.gz /data/projects/fate/proxy/fate-proxy-1.1.tar.gz

RUN cd /data/projects/fate/proxy/ && \
    tar -xzf fate-proxy-1.1.tar.gz && \
    rm fate-proxy-1.1.tar.gz && \
    ln -s fate-proxy-1.1.jar fate-proxy.jar

EXPOSE 9370

CMD cd /data/projects/fate/proxy && \
    java -cp "conf/:lib/*:fate-proxy.jar" com.webank.ai.fate.networking.Proxy -c conf/proxy.properties
