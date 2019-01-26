-- insertSelective database if not exists
CREATE DATABASE IF NOT EXISTS `eggroll_meta`;

USE `eggroll_meta`;
-- insertSelective table for fdn_meta

-- table
CREATE TABLE IF NOT EXISTS `eggroll_meta`.`dtable` (
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

CREATE UNIQUE INDEX `idx_u_table_n_tn` ON `eggroll_meta`.`dtable` (`namespace`(100), `table_name`(600), `status`(68));
CREATE INDEX `idx_table_np` ON `eggroll_meta`.`dtable` (`namespace`(768));
CREATE INDEX `idx_table_tn` ON `eggroll_meta`.`dtable` (`table_name`(768));
CREATE INDEX `idx_table_tt` ON `eggroll_meta`.`dtable` (`table_type`(255));
CREATE INDEX `idx_table_s` ON `eggroll_meta`.`dtable` (`status`(255));
CREATE INDEX `idx_table_sd` ON `eggroll_meta`.`dtable` (`serdes`(768));
CREATE INDEX `idx_table_sv` ON `eggroll_meta`.`dtable` (`storage_version`);


-- fragment
CREATE TABLE IF NOT EXISTS `eggroll_meta`.`fragment` (
  `fragment_id` SERIAL PRIMARY KEY,
  `table_id` BIGINT UNSIGNED NOT NULL,
  `node_id` BIGINT UNSIGNED NOT NULL,
  `fragment_order` INT UNSIGNED NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE UNIQUE INDEX `idx_u_fragment_ti_ni_fo` ON `eggroll_meta`.`fragment` (`table_id`, `node_id`, `fragment_order`);
CREATE INDEX `idx_fragment_ti` ON `eggroll_meta`.`fragment` (`table_id`);
CREATE INDEX `idx_fragment_ni` ON `eggroll_meta`.`fragment` (`node_id`);
CREATE INDEX `idx_fragment_s` ON `eggroll_meta`.`fragment` (`status`(255));


-- node
CREATE TABLE IF NOT EXISTS `eggroll_meta`.`node` (
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

CREATE INDEX `idx_node_h` ON `eggroll_meta`.`node` (`host`(768));
CREATE INDEX `idx_node_i` ON `eggroll_meta`.`node` (`ip`(255));
CREATE INDEX `idx_node_t` ON `eggroll_meta`.`node` (`type`(255));
CREATE INDEX `idx_node_s` ON `eggroll_meta`.`node` (`status`(255));


-- task
CREATE TABLE IF NOT EXISTS `eggroll_meta`.`task` (
  `task_id` SERIAL PRIMARY KEY,
  `task_name` VARCHAR(2000) NOT NULL,
  `task_type` VARCHAR(255) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE UNIQUE INDEX `idx_u_task_tn` ON `eggroll_meta`.`task` (`task_name`(768));
CREATE INDEX `idx_task_tt` ON `eggroll_meta`.`task` (`task_type`(255));
CREATE INDEX `idx_task_s` ON `eggroll_meta`.`task` (`status`(255));

-- result
CREATE TABLE IF NOT EXISTS `eggroll_meta`.`result` (
  `result_id` SERIAL PRIMARY KEY,
  `result_name` VARCHAR(2000),
  `result_type` VARCHAR(2000) NOT NULL,
  `value` VARCHAR(10000) NOT NULL,
  `status` VARCHAR(255) NOT NULL,
  `task_id` BIGINT UNSIGNED,
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
  `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE INDEX `idx_result_rn` ON `eggroll_meta`.`result` (`result_name`(768));
CREATE INDEX `idx_result_rt` ON `eggroll_meta`.`result` (`result_type`(768));
CREATE INDEX `idx_result_s` ON `eggroll_meta`.`result` (`status`(255));
CREATE INDEX `idx_result_ti` ON `eggroll_meta`.`result` (`task_id`);
