# This the base python file of get intersection output, should no modify it

from arch.api import eggroll

intersect_data_output_table = "guest_intersect_output_table_name"
intersect_data_output_namespace = "guest_intersect_output_namespace"
eggroll.init("get_intersect_output", mode = 1)
table = eggroll.table(intersect_data_output_table, intersect_data_output_namespace)

print("intersect output count:{}".format(table.count()))

### if you want to see the id in resuts, uncomment these code
# for data in list(table.collect()):
#    print("id:{}, value:{}".format(data[0], data[1]))
