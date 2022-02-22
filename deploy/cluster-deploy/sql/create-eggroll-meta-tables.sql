-- create database if not exists, default database is eggroll_meta
-- CREATE DATABASE IF NOT EXISTS `eggroll_meta`;

-- all operation under this database
-- USE `eggroll_meta`;

-- store_locator
CREATE TABLE IF NOT EXISTS `store_locator` (
  `store_locator_id` SERIAL PRIMARY KEY,
  `store_type` VARCHAR(255) NOT NULL,
  `namespace` VARCHAR(2000) NOT NULL DEFAULT 'DEFAULT',
  `name` VARCHAR(2000) NOT NULL,
  `path` VARCHAR(2000) NOT NULL DEFAULT '',
  `total_partitions` INT UNSIGNED NOT NULL,
  `partitioner` VARCHAR(2000) NOT NULL DEFAULT 'BYTESTRING_HASH',
  `serdes` VARCHAR(2000) NOT NULL DEFAULT '',
  `version` INT UNSIGNED NOT NULL DEFAULT 0,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE UNIQUE INDEX `idx_u_store_loinsert-node.sqlcator_ns_n` ON `store_locator` (`namespace`(120), `name`(640));
CREATE INDEX `idx_store_locator_st` ON `store_locator` (`store_type`(255));
CREATE INDEX `idx_store_locator_ns` ON `store_locator` (`namespace`(767));
CREATE INDEX `idx_store_locator_n` ON `store_locator` (`name`(767));
CREATE INDEX `idx_store_locator_s` ON `store_locator` (`status`(255));
CREATE INDEX `idx_store_locator_v` ON `store_locator` (`version`);


-- store (option)
CREATE TABLE IF NOT EXISTS `store_option` (
  `store_option_id` SERIAL PRIMARY KEY,
  `store_locator_id` BIGINT UNSIGNED NOT NULL,
  `name` VARCHAR(255) NOT NULL,
  `data` VARCHAR(2000) NOT NULL DEFAULT '',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE INDEX `idx_store_option_si` ON `store_option` (`store_locator_id`);


-- store_partition
CREATE TABLE IF NOT EXISTS `store_partition` (
  `store_partition_id` SERIAL PRIMARY KEY,          -- self-increment sequence
  `store_locator_id` BIGINT UNSIGNED NOT NULL,
  `node_id` BIGINT UNSIGNED NOT NULL,
  `partition_id` INT UNSIGNED NOT NULL,             -- partition id of a store
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE UNIQUE INDEX `idx_u_store_partition_si_spi_ni` ON `store_partition` (`store_locator_id`, `store_partition_id`, `node_id`);
CREATE INDEX `idx_store_partition_sli` ON `store_partition` (`store_locator_id`);
CREATE INDEX `idx_store_partition_ni` ON `store_partition` (`node_id`);
CREATE INDEX `idx_store_partition_s` ON `store_partition` (`status`(255));


-- node
CREATE TABLE IF NOT EXISTS `server_node` (
  `server_node_id` SERIAL PRIMARY KEY,
  `name` VARCHAR(2000) NOT NULL DEFAULT '',
  `server_cluster_id` BIGINT UNSIGNED NOT NULL DEFAULT 0,
  `host` VARCHAR(1000) NOT NULL,
  `port` INT NOT NULL,
  `node_type` VARCHAR(255) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `last_heartbeat_at` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE INDEX `idx_server_node_h_p_nt` ON `server_node` (`host`(600), `port`, `node_type`(100));
CREATE INDEX `idx_server_node_h` ON `server_node` (`host`(767));
CREATE INDEX `idx_server_node_sci` ON `server_node` (`server_cluster_id`);
CREATE INDEX `idx_server_node_nt` ON `server_node` (`node_type`(255));
CREATE INDEX `idx_server_node_s` ON `server_node` (`status`(255));


-- session (main)
CREATE TABLE IF NOT EXISTS `session_main` (
  `session_id` VARCHAR(767) PRIMARY KEY,
  `name` VARCHAR(2000) NOT NULL DEFAULT '',
  `status` VARCHAR(255) NOT NULL,
  `tag` VARCHAR(255),
  `total_proc_count` INT,
  `active_proc_count` INT,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE INDEX `idx_session_main_s` ON `session_main` (`status`);


-- session (option)
CREATE TABLE IF NOT EXISTS `session_option` (
  `session_option_id`  SERIAL PRIMARY KEY,
  `session_id` VARCHAR(2000),
  `name` VARCHAR(255) NOT NULL,
  `data` VARCHAR(2000) NOT NULL DEFAULT '',
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE INDEX `idx_session_option_si` ON `session_option` (`session_id`(767));


-- session (processor)
CREATE TABLE IF NOT EXISTS `session_processor` (
  `processor_id` SERIAL PRIMARY KEY,
  `session_id` VARCHAR(767),
  `server_node_id` INT NOT NULL,
  `processor_type` VARCHAR(255) NOT NULL,
  `status` VARCHAR(255),
  `tag` VARCHAR(255),
  `command_endpoint` VARCHAR(255),
  `transfer_endpoint` VARCHAR(255),
  `pid` INT NOT NULL DEFAULT -1,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;

CREATE INDEX `idx_session_processor_si` ON `session_processor` (`session_id`(767));
