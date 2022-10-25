ALTER TABLE t_machine_learning_model_info ADD f_archive_sha256 VARCHAR(100);
ALTER TABLE t_machine_learning_model_info ADD f_archive_from_ip VARCHAR(100);
ALTER TABLE t_task ADD f_run_port int DEFAULT NULL;
ALTER TABLE t_task ADD f_kill_status tinyint(1) default 1;
ALTER TABLE t_task ADD f_error_report text;