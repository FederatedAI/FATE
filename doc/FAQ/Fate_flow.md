1. **请问fate_flow需要在所有的节点上启动，还是只在host上就行?**
所有参与方均需要一个

2. **运行启动fate_flow服务得到以下结果:
service start sucessfully. pid: 13146
status: app       13146  0.0  0.0 188948 12476 pts/1    R+   02:02   0:00 python fate_flow_server.py
但是，再查看fate_flow状态，显示：service not running**
可以查看PYTHONPATH/logs/fate_flow/fate_flow_stat.log排查问题

3. **FATE flow提交任务的时候显示成功了，在dashboard页面任务却是失败状态?**
FATE flow的submit job只是提交任务，success代表提交成功，实际执行的任务失败了需要看日志。可以通过board来看日志
