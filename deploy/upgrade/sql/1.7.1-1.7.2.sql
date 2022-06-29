ALTER TABLE t_job DROP INDEX job_f_user_id;
ALTER TABLE t_job DROP INDEX job_f_tag;
ALTER TABLE t_job DROP INDEX job_f_initiator_role;
ALTER TABLE t_job DROP INDEX job_f_initiator_party_id;
ALTER TABLE t_job DROP INDEX job_f_status;
ALTER TABLE t_job DROP INDEX job_f_status_code;
ALTER TABLE t_job DROP INDEX job_f_is_initiator;
ALTER TABLE t_job DROP INDEX job_f_ready_signal;
ALTER TABLE t_job DROP INDEX job_f_cancel_signal;
ALTER TABLE t_job DROP INDEX job_f_rerun_signal;
ALTER TABLE t_job DROP INDEX job_f_engine_name;
ALTER TABLE t_job DROP INDEX job_f_engine_type;
ALTER TABLE t_job DROP INDEX job_f_cores;
ALTER TABLE t_job DROP INDEX job_f_memory;
ALTER TABLE t_job DROP INDEX job_f_remaining_cores;
ALTER TABLE t_job DROP INDEX job_f_remaining_memory;
ALTER TABLE t_job DROP INDEX job_f_resource_in_use;

ALTER TABLE t_task DROP INDEX task_f_component_module;
ALTER TABLE t_task DROP INDEX task_f_task_id;
ALTER TABLE t_task DROP INDEX task_f_task_version;
ALTER TABLE t_task DROP INDEX task_f_initiator_role;
ALTER TABLE t_task DROP INDEX task_f_initiator_party_id;
ALTER TABLE t_task DROP INDEX task_f_federated_mode;
ALTER TABLE t_task DROP INDEX task_f_federated_status_collect_type;
ALTER TABLE t_task DROP INDEX task_f_status_code;
ALTER TABLE t_task DROP INDEX task_f_auto_retries;
ALTER TABLE t_task DROP INDEX task_f_worker_id;
ALTER TABLE t_task DROP INDEX task_f_party_status;

ALTER TABLE trackingmetric DROP INDEX trackingmetric_f_metric_namespace;
ALTER TABLE trackingmetric DROP INDEX trackingmetric_f_metric_name;
ALTER TABLE trackingmetric DROP INDEX trackingmetric_f_type;

ALTER TABLE trackingoutputdatainfo DROP INDEX trackingoutputdatainfo_f_task_version;

ALTER TABLE t_machine_learning_model_info DROP INDEX machinelearningmodelinfo_f_role;
ALTER TABLE t_machine_learning_model_info DROP INDEX machinelearningmodelinfo_f_party_id;
ALTER TABLE t_machine_learning_model_info DROP INDEX machinelearningmodelinfo_f_initiator_role;
ALTER TABLE t_machine_learning_model_info DROP INDEX machinelearningmodelinfo_f_initiator_party_id;

ALTER TABLE t_data_table_tracking DROP INDEX datatabletracking_f_table_name;
ALTER TABLE t_data_table_tracking DROP INDEX datatabletracking_f_table_namespace;

ALTER TABLE t_cache_record DROP PRIMARY KEY, ADD id INT NOT NULL AUTO_INCREMENT PRIMARY KEY FIRST;
ALTER TABLE t_cache_record DROP INDEX cacherecord_f_task_id;

DROP PROCEDURE IF EXISTS alter_componentsummary;
DELIMITER //

CREATE PROCEDURE alter_componentsummary()
BEGIN
    DECLARE done BOOL DEFAULT FALSE;
    DECLARE date_ CHAR(8);

    DECLARE cur CURSOR FOR SELECT RIGHT(TABLE_NAME, 8) FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = (SELECT DATABASE()) AND TABLE_NAME LIKE 't\_component\_summary\_%';
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    OPEN cur;

    loop_: LOOP
        FETCH cur INTO date_;
        IF done THEN
            LEAVE loop_;
        END IF;

        SET @sql = CONCAT('ALTER TABLE t_component_summary_', date_, ' DROP INDEX componentsummary_', date_, '_f_task_version');
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END LOOP;

    CLOSE cur;
END //

DELIMITER ;
CALL alter_componentsummary();
DROP PROCEDURE alter_componentsummary;

ALTER TABLE t_model_operation_log DROP INDEX modeloperationlog_f_model_id;
ALTER TABLE t_model_operation_log DROP INDEX modeloperationlog_f_model_version;

ALTER TABLE t_engine_registry DROP INDEX engineregistry_f_cores;
ALTER TABLE t_engine_registry DROP INDEX engineregistry_f_memory;
ALTER TABLE t_engine_registry DROP INDEX engineregistry_f_remaining_cores;
ALTER TABLE t_engine_registry DROP INDEX engineregistry_f_remaining_memory;
ALTER TABLE t_engine_registry DROP INDEX engineregistry_f_nodes;

ALTER TABLE t_worker DROP INDEX workerinfo_f_task_id;
ALTER TABLE t_worker DROP INDEX workerinfo_f_role;

CREATE TABLE t_storage_connector (
    f_name VARCHAR(100) NOT NULL,
    f_create_time BIGINT,
    f_create_date DATETIME,
    f_update_time BIGINT,
    f_update_date DATETIME,
    f_engine VARCHAR(100) NOT NULL,
    f_connector_info LONGTEXT NOT NULL,

    PRIMARY KEY (f_name),
    INDEX storageconnectormodel_f_engine (f_engine)
);

ALTER TABLE t_storage_table_meta DROP INDEX storagetablemetamodel_f_engine;
ALTER TABLE t_storage_table_meta DROP INDEX storagetablemetamodel_f_store_type;

ALTER TABLE t_session_record DROP INDEX sessionrecord_f_engine_session_id;
ALTER TABLE t_session_record DROP INDEX sessionrecord_f_manager_session_id;
