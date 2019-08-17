#base image
FROM mysql:latest

#create database
COPY ./init_database.sql /docker-entrypoint-initdb.d

#add execute pri
RUN chmod a+x /docker-entrypoint-initdb.d/init_database.sql
