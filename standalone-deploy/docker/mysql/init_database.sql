CREATE DATABASE IF NOT EXISTS fate_flow;
ALTER USER 'root'@'%' IDENTIFIED WITH mysql_native_password BY 'fate_dev';
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'fate_dev';
CREATE USER 'fate_dev'@'%' IDENTIFIED WITH mysql_native_password BY 'fate_dev';
GRANT ALL ON *.* TO 'fate_dev'@'%';
flush privileges;
