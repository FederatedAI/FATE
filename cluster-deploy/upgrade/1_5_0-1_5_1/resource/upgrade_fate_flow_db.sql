USE fate_flow

CREATE TABLE t_job_backup150 LIKE t_job;
INSERT INTO t_job_backup150 SELECT * FROM t_job;
CREATE TABLE t_machine_learning_model_info_backup150 LIKE t_machine_learning_model_info;
INSERT INTO t_machine_learning_model_info_backup150 SELECT * FROM t_machine_learning_model_info;

ALTER TABLE t_job CHANGE f_cancel_time f_cancel_time bigint(20) DEFAULT NULL;
ALTER TABLE t_job ADD f_end_scheduling_updates int(11) DEFAULT 0;

ALTER TABLE t_machine_learning_model_info CHANGE f_dsl f_train_dsl longtext NOT NULL;
ALTER TABLE t_machine_learning_model_info CHANGE f_job_status f_job_status varchar(50);
ALTER TABLE t_machine_learning_model_info ADD f_runtime_conf_on_party longtext NOT NULL;
ALTER TABLE t_machine_learning_model_info ADD f_fate_version varchar(10) NOT NULL;
ALTER TABLE t_machine_learning_model_info ADD f_parent tinyint(1) DEFAULT NULL;
ALTER TABLE t_machine_learning_model_info ADD f_parent_info longtext NOT NULL;
ALTER TABLE t_machine_learning_model_info ADD f_inference_dsl longtext NOT NULL;
ALTER TABLE t_machine_learning_model_info DROP COLUMN f_id;
ALTER TABLE t_machine_learning_model_info ADD PRIMARY KEY (f_role, f_party_id, f_model_id, f_model_version);

ALTER TABLE t_model_tag CHANGE f_m_id f_m_id varchar(25);