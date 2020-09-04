use fate_flow;
alter table t_job modify f_role varchar(50);
alter table t_task modify f_role varchar(50);
alter table t_data_view modify f_role varchar(50);
alter table t_machine_learning_model_meta modify f_role varchar(50);