FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine

WORKDIR /data/projects/fate/federation/

COPY fate-federation-*.tar.gz ./
COPY conf ./conf

RUN tar -xzf fate-federation-*.tar.gz && \
    rm fate-federation-*.tar.gz && \
    ln -s fate-federation-*.jar fate-federation.jar
EXPOSE 9394

CMD java -cp "conf/:lib/*:fate-federation.jar" com.webank.ai.fate.driver.Federation -c conf/federation.properties

