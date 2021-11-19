FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine as builder
WORKDIR /data/projects/fate/
COPY fateboard.tar.gz .
RUN tar -xzf fateboard.tar.gz 

FROM mcr.microsoft.com/java/jre:8u192-zulu-alpine
RUN apk add tzdata
WORKDIR /data/projects/fate/fateboard/

COPY --from=builder /data/projects/fate/fateboard /data/projects/fate/fateboard
EXPOSE 8080

CMD java -Dspring.config.location=/data/projects/fate/fateboard/conf/application.properties  -Dssh_config_file=/data/projects/fate/fateboard/conf  -Xmx2048m -Xms2048m -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log -XX:+HeapDumpOnOutOfMemoryError  -jar fateboard.jar