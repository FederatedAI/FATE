ALTER TABLE t_job ADD f_inheritance_info LONGTEXT;
ALTER TABLE t_job ADD f_inheritance_status VARCHAR(50);

ALTER TABLE t_storage_table_meta MODIFY f_count BIGINT;
