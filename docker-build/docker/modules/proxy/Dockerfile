FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

RUN mkdir -p /data/projects/fate
COPY . /data/projects/fate/proxy
EXPOSE 9370

CMD cd /data/projects/fate/proxy && \
    java -cp "conf/:lib/*:fate-proxy.jar" com.webank.ai.fate.networking.Proxy -c conf/proxy.properties
