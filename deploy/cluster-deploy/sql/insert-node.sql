CREATE DATABASE IF NOT EXISTS fate_flow;
CREATE USER fate@'localhost' IDENTIFIED BY 'fate';
GRANT ALL ON fate_flow.* TO fate@'localhost';
GRANT ALL ON eggroll_meta.* TO fate@'localhost;
use eggroll_meta;
INSERT INTO server_node (host, port, node_type, status) values ('127.0.0.1', '4670', 'CLUSTER_MANAGER', 'HEALTHY');
INSERT INTO server_node (host, port, node_type, status) values ('127.0.0.1', '4671', 'NODE_MANAGER', 'HEALTHY');
