ALTER TABLE t_task ADD f_world_info LONGTEXT default null;
ALTER TABLE t_task ADD f_launcher VARCHAR(20) default null;
ALTER TABLE t_task ADD f_master_addr VARCHAR(30) default null;
ALTER TABLE t_engine_registry ADD f_devices INTEGER default 0;
ALTER TABLE t_engine_registry ADD f_remaining_devices INTEGER default NULL;