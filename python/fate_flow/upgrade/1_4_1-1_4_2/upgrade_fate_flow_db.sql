use fate_flow;

create table t_job_backup like t_job;
insert into t_job_backup select * from t_job;
alter table t_job modify f_role varchar(50);

create table t_task_backup like t_task;
insert into t_task_backup select * from t_task;
alter table t_task modify f_role varchar(50);

create table t_data_view_backup like t_data_view;
insert into t_data_view_backup select * from t_data_view;
alter table t_data_view modify f_role varchar(50);

create table t_machine_learning_model_meta_backup like t_machine_learning_model_meta;
insert into t_machine_learning_model_meta_backup select * from t_machine_learning_model_meta;
alter table t_machine_learning_model_meta modify f_role varchar(50);