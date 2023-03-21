ALTER TABLE t_job ADD f_user LONGTEXT NOT NULL;
UPDATE t_job SET f_user = '{}' WHERE f_user = '';

ALTER TABLE t_task ADD f_engine_conf LONGTEXT;
