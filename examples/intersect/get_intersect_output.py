# This the base python file of get intersection output, should no modify it

from arch.api import eggroll

name = "_intersect_output_table_name"
namespace = "_intersect_output_namespace"

eggroll.init("get_intersect_output", mode = _work_mode)
table = eggroll.table(name, namespace)

print("intersect output count:"+str(table.count()))
