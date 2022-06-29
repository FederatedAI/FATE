ALTER TABLE t_job DROP f_work_mode;

ALTER TABLE t_task ADD f_component_module VARCHAR(200) NOT NULL, ADD INDEX task_f_component_module (f_component_module);
ALTER TABLE t_task ADD f_auto_retries INT NOT NULL DEFAULT 0, ADD INDEX task_f_auto_retries (f_auto_retries);
ALTER TABLE t_task ADD f_auto_retry_delay INT NOT NULL DEFAULT 0;
ALTER TABLE t_task ADD f_worker_id VARCHAR(100), ADD INDEX task_f_worker_id (f_worker_id);
ALTER TABLE t_task ADD f_cmd LONGTEXT;
ALTER TABLE t_task ADD f_provider_info LONGTEXT NOT NULL;
UPDATE t_task SET f_provider_info = '{}' WHERE f_provider_info = '';
ALTER TABLE t_task ADD f_component_parameters LONGTEXT NOT NULL;
UPDATE t_task SET f_component_parameters = '{}' WHERE f_component_parameters = '';

ALTER TABLE t_machine_learning_model_info DROP f_work_mode;

CREATE TABLE t_data_table_tracking (
    f_table_id BIGINT NOT NULL AUTO_INCREMENT,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_table_name VARCHAR(300),
    f_table_namespace VARCHAR(300),
    f_job_id VARCHAR(25),
    f_have_parent BOOL NOT NULL DEFAULT FALSE,
    f_parent_number INT NOT NULL DEFAULT 0,
    f_parent_table_name VARCHAR(500),
    f_parent_table_namespace VARCHAR(500),
    f_source_table_name VARCHAR(500),
    f_source_table_namespace VARCHAR(500),

    PRIMARY KEY (f_table_id),
    INDEX datatabletracking_f_table_name (f_table_name),
    INDEX datatabletracking_f_table_namespace (f_table_namespace),
    INDEX datatabletracking_f_job_id (f_job_id)
);

CREATE TABLE t_cache_record (
    f_cache_key VARCHAR(500) NOT NULL,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_cache LONGTEXT NOT NULL,
    f_job_id VARCHAR(25),
    f_role VARCHAR(50),
    f_party_id VARCHAR(10),
    f_component_name TEXT,
    f_task_id VARCHAR(100),
    f_task_version BIGINT,
    f_cache_name VARCHAR(50),
    t_ttl BIGINT NOT NULL DEFAULT 0,

    PRIMARY KEY (f_cache_key),
    INDEX cacherecord_f_job_id (f_job_id),
    INDEX cacherecord_f_role (f_role),
    INDEX cacherecord_f_party_id (f_party_id),
    INDEX cacherecord_f_task_id (f_task_id),
    INDEX cacherecord_f_task_version (f_task_version)
);

CREATE TABLE t_component_registry (
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_provider_name VARCHAR(20) NOT NULL,
    f_version VARCHAR(10) NOT NULL,
    f_component_name VARCHAR(30) NOT NULL,
    f_module VARCHAR(128) NOT NULL,

    PRIMARY KEY (f_provider_name, f_version, f_component_name),
    INDEX componentregistryinfo_f_provider_name (f_provider_name),
    INDEX componentregistryinfo_f_version (f_version),
    INDEX componentregistryinfo_f_component_name (f_component_name)
);

CREATE TABLE t_component_provider_info (
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_provider_name VARCHAR(20) NOT NULL,
    f_version VARCHAR(10) NOT NULL,
    f_class_path LONGTEXT NOT NULL,
    f_path VARCHAR(128) NOT NULL,
    f_python VARCHAR(128) NOT NULL,

    PRIMARY KEY (f_provider_name, f_version),
    INDEX componentproviderinfo_f_provider_name (f_provider_name),
    INDEX componentproviderinfo_f_version (f_version)
);

CREATE TABLE t_component_info (
    f_component_name VARCHAR(30) NOT NULL,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_component_alias LONGTEXT NOT NULL,
    f_default_provider VARCHAR(20) NOT NULL,
    f_support_provider LONGTEXT,

    PRIMARY KEY (f_component_name)
);

CREATE TABLE t_worker (
    f_worker_id VARCHAR(100) NOT NULL,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_worker_name VARCHAR(50) NOT NULL,
    f_job_id VARCHAR(25) NOT NULL,
    f_task_id VARCHAR(100) NOT NULL,
    f_task_version BIGINT NOT NULL,
    f_role VARCHAR(50) NOT NULL,
    f_party_id VARCHAR(10) NOT NULL,
    f_run_ip VARCHAR(100),
    f_run_pid INT,
    f_http_port INT,
    f_grpc_port INT,
    f_config LONGTEXT,
    f_cmd LONGTEXT,
    f_start_time BIGINT,
    f_start_date DATETIME,
    f_end_time BIGINT,
    f_end_date DATETIME,

    PRIMARY KEY (f_worker_id),
    INDEX workerinfo_f_worker_name (f_worker_name),
    INDEX workerinfo_f_job_id (f_job_id),
    INDEX workerinfo_f_task_id (f_task_id),
    INDEX workerinfo_f_task_version (f_task_version),
    INDEX workerinfo_f_role (f_role),
    INDEX workerinfo_f_party_id (f_party_id)
);

CREATE TABLE t_dependencies_storage_meta (
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_storage_engine VARCHAR(30) NOT NULL,
    f_type VARCHAR(20) NOT NULL,
    f_version VARCHAR(10) NOT NULL,
    f_storage_path VARCHAR(256),
    f_snapshot_time BIGINT,
    f_fate_flow_snapshot_time BIGINT,
    f_dependencies_conf LONGTEXT,
    f_upload_status BOOL NOT NULL DEFAULT FALSE,
    f_pid INT,

    PRIMARY KEY (f_storage_engine, f_type, f_version),
    INDEX dependenciesstoragemeta_f_version (f_version)
);

ALTER TABLE t_storage_table_meta CHANGE f_type f_store_type VARCHAR(50), RENAME INDEX storagetablemetamodel_f_type TO storagetablemetamodel_f_store_type;
ALTER TABLE t_storage_table_meta ADD f_extend_sid BOOL NOT NULL DEFAULT FALSE;
ALTER TABLE t_storage_table_meta ADD f_auto_increasing_sid BOOL NOT NULL DEFAULT FALSE;
ALTER TABLE t_storage_table_meta ADD f_read_access_time BIGINT;
ALTER TABLE t_storage_table_meta ADD f_read_access_date DATETIME;
ALTER TABLE t_storage_table_meta ADD f_write_access_time BIGINT;
ALTER TABLE t_storage_table_meta ADD f_write_access_date DATETIME;
ALTER TABLE t_storage_table_meta MODIFY f_create_time BIGINT;

ALTER TABLE t_session_record RENAME COLUMN f_session_id TO f_engine_session_id, ADD INDEX sessionrecord_f_engine_session_id (f_engine_session_id);
ALTER TABLE t_session_record ADD f_manager_session_id VARCHAR(150) NOT NULL, ADD INDEX sessionrecord_f_manager_session_id (f_manager_session_id);
ALTER TABLE t_session_record MODIFY f_create_time BIGINT, DROP INDEX sessionrecord_f_create_time;
ALTER TABLE t_session_record DROP PRIMARY KEY, ADD PRIMARY KEY (f_engine_type, f_engine_name, f_engine_session_id);
