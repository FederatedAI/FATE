use fate_flow;

create table t_queue_backup like t_queue;
insert into t_queue_backup select * from t_queue;
alter table t_queue add f_frequency int default 0;
