1. **fate flow在fate中充当的角色是什么?**
FATE Flow是调度系统，根据用户提交的作业DSL，调度算法组件执行。

2. **启动服务时出现ModuleNotFoundError: No module named 'arch'**
将PYTHONPATH设置为fate_flow的上层目录。

3. **为什么提交任务的时候显示成功了，在dashboard页面任务却是失败状态？**
submit job只是提交任务，success代表提交成功，job失败需要看日志。可以通过board来看日志

4. **在fate中  guest、host、arbiter、local  角色分别代表了什么含义及作用?**
arbiter是用来辅助多方完成联合建模的，他主要的作用是用来聚合梯度或者模型。比如纵向lr里面，各方将自己一半的梯度发送给arbiter，然后arbiter再联合优化等等;
Guest表示数据应用方;
Host是数据提供方，在纵向算法中，Guest往往是有y的一方;
Local是指本地，只对upload和download有效。

5. **为什么无法kill掉的waiting jobs，kill的时候显示"cannot find"?**
Fate_flow目前仅支持在job发起方进行kill，其他方kill会显示cannot find

6. **upload data 这一步是在做什么？**
Upload data是上传到eggroll里面，变成后续算法可执行的DTable格式

7. **请问算法中间的产生数据怎么查看？**
可以使用 python fate_flow_client.py -f component_output_model -j $job_id -r $role -g $guest -cpn $component_name -o $output_path

8. **如果同一个文件upload执行两遍，fate是会删掉第一次的数据，重新上传吗?**
如果同一表中的键相同，它将被覆盖。

9. **某个job失败而fateboard上并没有错误日志的原因是什么？**
这几个地方的日志不会展示在board上: 对应job下的fate_flow_schedule.log， logs/error.log,   logs/fate_flow/ERROR.log

10. **load和bind这两个命令有什么区别？**
load可以理解为模型发布，bind是设置默认的模型版本。

11. **请问fate_flow需要在所有的节点上启动，还是只在host上就行?**
所有参与方均需要一个

