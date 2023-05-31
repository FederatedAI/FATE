CREATE DATABASE IF NOT EXISTS fate_flow;
CREATE USER fate@'localhost' IDENTIFIED BY 'fate';
GRANT ALL ON fate_flow.* TO fate@'localhost';
GRANT ALL ON eggroll_meta.* TO fate@'localhost;
