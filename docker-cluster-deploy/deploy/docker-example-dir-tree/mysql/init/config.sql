-- insertSelective database if not exists
CREATE DATABASE IF NOT EXISTS `fate`;

USE `fate`;
-- insertSelective table for fdn_meta

-- table
CREATE TABLE IF NOT EXISTS `fate`.`dtable` (
  `table_id` SERIAL PRIMARY KEY,
  `namespace` VARCHAR(2000) NOT NULL DEFAULT 'DEFAULT',
  `table_name` VARCHAR(2000) NOT NULL,
  `table_type` VARCHAR(255) NOT NULL,
  `total_fragments` INT UNSIGNED NOT NULL,
  `dispatcher` VARCHAR(2000) NOT NULL DEFAULT 'DEFAULT',
  `serdes` VARCHAR(2000) NOT NULL DEFAULT 'MOD',
  `storage_version` INT UNSIGNED NOT NULL DEFAULT 0,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE UNIQUE INDEX `idx_u_table_n_tn` ON `fate`.`dtable` (`namespace`(100), `table_name`(600), `status`(68));
CREATE INDEX `idx_table_np` ON `fate`.`dtable` (`namespace`(768));
CREATE INDEX `idx_table_tn` ON `fate`.`dtable` (`table_name`(768));
CREATE INDEX `idx_table_tt` ON `fate`.`dtable` (`table_type`(255));
CREATE INDEX `idx_table_s` ON `fate`.`dtable` (`status`(255));
CREATE INDEX `idx_table_sd` ON `fate`.`dtable` (`serdes`(768));
CREATE INDEX `idx_table_sv` ON `fate`.`dtable` (`storage_version`);


-- fragment
CREATE TABLE IF NOT EXISTS `fate`.`fragment` (
  `fragment_id` SERIAL PRIMARY KEY,
  `table_id` BIGINT UNSIGNED NOT NULL,
  `node_id` BIGINT UNSIGNED NOT NULL,
  `fragment_order` INT UNSIGNED NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE UNIQUE INDEX `idx_u_fragment_ti_ni_fo` ON `fate`.`fragment` (`table_id`, `node_id`, `fragment_order`);
CREATE INDEX `idx_fragment_ti` ON `fate`.`fragment` (`table_id`);
CREATE INDEX `idx_fragment_ni` ON `fate`.`fragment` (`node_id`);
CREATE INDEX `idx_fragment_s` ON `fate`.`fragment` (`status`(255));


-- node
CREATE TABLE IF NOT EXISTS `fate`.`node` (
  `node_id` SERIAL PRIMARY KEY,
  `host` VARCHAR(1000),
  `ip` VARCHAR(255) NOT NULL,
  `port` INT NOT NULL,
  `type` VARCHAR(255) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `last_heartbeat_at` DATETIME DEFAULT NULL ON UPDATE CURRENT_TIMESTAMP,
  `created_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE INDEX `idx_node_h` ON `fate`.`node` (`host`(768));
CREATE INDEX `idx_node_i` ON `fate`.`node` (`ip`(255));
CREATE INDEX `idx_node_t` ON `fate`.`node` (`type`(255));
CREATE INDEX `idx_node_s` ON `fate`.`node` (`status`(255));


-- task
CREATE TABLE IF NOT EXISTS `fate`.`task` (
  `task_id` SERIAL PRIMARY KEY,
  `task_name` VARCHAR(2000) NOT NULL,
  `task_type` VARCHAR(255) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE UNIQUE INDEX `idx_u_task_tn` ON `fate`.`task` (`task_name`(768));
CREATE INDEX `idx_task_tt` ON `fate`.`task` (`task_type`(255));
CREATE INDEX `idx_task_s` ON `fate`.`task` (`status`(255));

-- result
CREATE TABLE IF NOT EXISTS `fate`.`result` (
  `result_id` SERIAL PRIMARY KEY,
  `result_name` VARCHAR(2000),
  `result_type` VARCHAR(2000) NOT NULL,
  `value` VARCHAR(10000) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `task_id` BIGINT UNSIGNED,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE INDEX `idx_result_rn` ON `fate`.`result` (`result_name`(768));
CREATE INDEX `idx_result_rt` ON `fate`.`result` (`result_type`(768));
CREATE INDEX `idx_result_s` ON `fate`.`result` (`status`(255));
CREATE INDEX `idx_result_ti` ON `fate`.`result` (`task_id`);





use fate;
INSERT INTO node (ip, port, type, status) values ('roll', '8011', 'ROLL', 'HEALTHY');
INSERT INTO node (ip, port, type, status) values ('proxy', '9370', 'PROXY', 'HEALTHY');

INSERT INTO node (ip, port, type, status) values ('egg', '7888', 'EGG', 'HEALTHY');
INSERT INTO node (ip, port, type, status) values ('egg', '7778', 'STORAGE', 'HEALTHY');
