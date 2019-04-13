# This the base python file of get intersection output, should no modify it

from arch.api import eggroll

name = "_intersect_output_table_name"
namespace = "_intersect_output_namespace"
role = name.split('_')[0]
eggroll.init("get_intersect_output", mode = _work_mode)
table = eggroll.table(name, namespace)

print(role +"_intersect output count:"+str(table.count()))
