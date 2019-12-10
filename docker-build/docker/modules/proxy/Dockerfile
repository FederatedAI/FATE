FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

WORKDIR /data/projects/fate/proxy/

COPY fate-proxy-*.tar.gz ./
COPY conf ./conf

RUN tar -xzf fate-proxy-*.tar.gz && \
    rm fate-proxy-*.tar.gz && \
    ln -s fate-proxy-*.jar fate-proxy.jar

EXPOSE 9370

CMD java -cp "conf/:lib/*:fate-proxy.jar" com.webank.ai.fate.networking.Proxy -c conf/proxy.properties
