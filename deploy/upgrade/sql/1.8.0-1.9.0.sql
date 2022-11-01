ALTER TABLE t_task ADD f_run_port INT;
ALTER TABLE t_task ADD f_kill_status BOOL NOT NULL DEFAULT FALSE;
ALTER TABLE t_task ADD f_error_report TEXT;

ALTER TABLE t_machine_learning_model_info DROP f_description;
ALTER TABLE t_machine_learning_model_info DROP f_job_status;
ALTER TABLE t_machine_learning_model_info ADD f_archive_sha256 VARCHAR(100);
ALTER TABLE t_machine_learning_model_info ADD f_archive_from_ip VARCHAR(100);

DROP TABLE t_model_operation_log;

CREATE TABLE t_server_registry_info (
    id INT NOT NULL AUTO_INCREMENT,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_server_name VARCHAR(30) NOT NULL,
    f_host VARCHAR(30) NOT NULL,
    f_port INT NOT NULL,
    f_protocol VARCHAR(10) NOT NULL,

    PRIMARY KEY (id),
    INDEX serverregistryinfo_f_server_name (f_server_name)
);

CREATE TABLE t_service_registry_info (
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_server_name VARCHAR(30) NOT NULL,
    f_service_name VARCHAR(30) NOT NULL,
    f_url VARCHAR(100) NOT NULL,
    f_method VARCHAR(10) NOT NULL,
    f_params LONGTEXT,
    f_data LONGTEXT,
    f_headers LONGTEXT,

    PRIMARY KEY (f_server_name, f_service_name)
);

CREATE TABLE t_site_key_info (
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_party_id VARCHAR(10) NOT NULL,
    f_key_name VARCHAR(10) NOT NULL,
    f_key LONGTEXT NOT NULL,

    PRIMARY KEY (f_party_id, f_key_name)
);

CREATE TABLE t_pipeline_component_meta (
    id INT NOT NULL AUTO_INCREMENT,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_model_id VARCHAR(100) NOT NULL,
    f_model_version VARCHAR(100) NOT NULL,
    f_role VARCHAR(50) NOT NULL,
    f_party_id VARCHAR(10) NOT NULL,
    f_component_name VARCHAR(100) NOT NULL,
    f_component_module_name VARCHAR(100) NOT NULL,
    f_model_alias VARCHAR(100) NOT NULL,
    f_model_proto_index LONGTEXT,
    f_run_parameters LONGTEXT,
    f_archive_sha256 VARCHAR(100),
    f_archive_from_ip VARCHAR(100),

    PRIMARY KEY (id),
    INDEX pipelinecomponentmeta_f_model_id (f_model_id),
    INDEX pipelinecomponentmeta_f_model_version (f_model_version),
    INDEX pipelinecomponentmeta_f_role (f_role),
    INDEX pipelinecomponentmeta_f_party_id (f_party_id),
    INDEX pipelinecomponentmeta_f_component_name (f_component_name),
    INDEX pipelinecomponentmeta_f_model_alias (f_model_alias),
    UNIQUE INDEX (f_model_id, f_model_version, f_role, f_party_id, f_component_name)
);
