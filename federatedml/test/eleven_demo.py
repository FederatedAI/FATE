#coding=utf-8
from arch.api import eggroll as eggroll
from arch.api import federation
from phe import paillier
from operator import add
import functools

'''
# 测试包括
# eggroll : init shutdown table parallelize
# storeage: save_as count collect  put  put_if_absent put_all get destroy range delete
# computation: map mapValues reduce mapPartitions join sample glom
# federation: send recv
'''

# eggroll : init shutdown table parallelize
def test_eggroll_init():
    try:
        print ("start eggroll init ")
        eggroll.init()
        print ("eggroll init success")
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_eggroll_shutdown():
    try:
        print ("start eggroll shutdown")
        eggroll.shutdown()
        print ("eggroll shutdown success")
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_eggroll_table():
    try:
        print ("start eggroll table")
        test_save_as_small_data()
        print ("eggroll shutdown success")
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_save_as_small_data():
    data1 = range(100)
    table1 = eggroll.parallelize(data1)
    table1.save_as("small_table", "namespace1")
    small_table = eggroll.table("small_table", "namespace1")
    table_list1 = list(small_table.collect())
    return (len(table_list1), len(data1))


# storeage: save_as count collect  put  put_if_absent put_all get destroy range delete
def test_save_as():
    try:
        data1 = range(100)
        table1 = eggroll.parallelize(data1)
        table1.save_as("small_table", "namespace1")
        return "save_as success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_count():
    try:
        count = 0
        table = eggroll.parallelize(range(10), partition=1)
        print("count:" + str(table.count()))
        return "count success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_collect():
    try:
        count = 0
        table = eggroll.parallelize(range(10), partition=1)
        for k, v in table.collect():
            count += 1
        print("collect:" + str(count))
        print("count:" + str(table.count()))
        return "collect success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_put():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        value = table1.put("c", ["eleven"])
        print(value)
        return (table1.count(), len(data1))
    except Exception as err:
        print ("occur err,{0}".format(err))
    
def test_put_if_absent():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        value = table1.put_if_absent("c", ["eleven"])
        print(value)
        value = table1.put_if_absent("c", ["eleven"])
        print(value)
        return (table1.count(), len(data1))
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_put_all():
    try:
        print (test_put_all_not_repeated_key())
        print (test_put_all_repeated_key())
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_put_all_not_repeated_key():
    data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
    data2 = [("e", ["tomato", "potato"])]
    table1 = eggroll.parallelize(data1, include_key=True, partition=1)
    table2 = eggroll.parallelize(data2, include_key=True, partition=1)
    value = table1.put_all(table2.collect())
    print(value)
    data1.extend(data2)
    return (table1.count(), len(data1))


def test_put_all_repeated_key():
    data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
    data2 = [("a", ["tomato", "potato"]), ("c", [1, 2, 3])]
    table1 = eggroll.parallelize(data1, include_key=True, partition=1)
    table2 = eggroll.parallelize(data2, include_key=True, partition=1)
    value = table1.put_all(table2.collect())
    print(value)
    # compare the count of the new table with the old
    key1_set = set(list(zip(*data1))[0])
    key2_set = set(list(zip(*data2))[0])
    intersection_set = key1_set.intersection(key2_set)
    data1_dict = dict(data1)
    for i in intersection_set:
        del data1_dict[i]
    data1 = list(data1_dict)
    data1.extend(data2)
    return (len(data1), table1.count())


def test_get():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        print (table1.get("a"))
        return "get success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_destroy():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        table1.destroy()
        return "destroy success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_range():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        print (table1.range("", "z"))
        return "range success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_delete():
    try:
        data1 = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
        table1 = eggroll.parallelize(data1, include_key=True, partition=1)
        print (table1.delete("z"))
        print (table1.delete("a"))
        return "delete success"
    except Exception as err:
        print ("occur err,{0}".format(err))


# computation: map mapValues reduce mapPartitions join sample glom
def test_map():
    try:
        table = eggroll.parallelize(["a","b","c"])
        for i in table.map(lambda k,v:(v,v+"1")).collect():
            print (i)
        return "map success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_mapValues():
    try:
        table = eggroll.parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])], include_key=True)
        for i in table.mapValues(lambda x: len(x)).collect():
            print (i)
        return "mapValues success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_reduce():
    try:
        test_reduce_global_var()
        test_reuce_variable_parameter()
        test_reduce_diff_location_parameter()
        test_reduce_location_parameter()
        test_reduce_one_record()
        return "reduce success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_reduce_one_record():
    one_record_reduce_value = eggroll.parallelize([1]).reduce(add)
    print (one_record_reduce_value, 1)


def _param_test(x, y, z):
    return (len(x) + 1) * (len(y) + 2) * (len(z) + 3)


def test_reduce_location_parameter():
    data = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
    table = eggroll.parallelize(data, include_key=True, partition=1)
    f = functools.partial(_param_test, [5])
    value = table.reduce(f)
    print (value, 40)


def test_reduce_diff_location_parameter():
    data = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
    table = eggroll.parallelize(data, include_key=True, partition=1)
    f = functools.partial(_param_test, [1, 2, "3", "222"])
    value = table.reduce(f)
    print (value, 100)


def _var_param_test(x, y, c=4, w=1, b=-1):
    return [x, y, c, w, b]


def test_reuce_variable_parameter():
    data = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"]), ("c", ["tomato"])]
    table = eggroll.parallelize(data, include_key=True, partition=1)
    f = functools.partial(_var_param_test, w=[1, "2", "3"])
    value = table.reduce(f)
    vs_value = [[['apple', 'banana', 'lemon'], ['grapes'], 4, [1, '2', '3'], -1], ['tomato'], 4, [1, '2', '3'], -1]
    print (value, vs_value)


def _global_test(x, y, z):
    return (len(x) + m) * (len(y) + n) * (len(z) + k)


def test_reduce_global_var():
    data = [("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])]
    table = eggroll.parallelize(data, include_key=True, partition=1)
    global m, n, k
    m = 1;
    n = 5;
    k = 2
    f = functools.partial(_global_test, [1, 2, "3", "222"])
    value1 = table.reduce(f)
    print (value1, 120)

    m = 1;
    n = 2;
    k = 2
    f = functools.partial(_global_test, [1, 2, "3", "222"])
    value2 = table.reduce(f)
    print (value2, 75)


def test_mapPartitions():
    try:
        test_mapPartition()
        #test_mapPartitions_one_partition()
        #test_mapPartitions_small_partition()
        #test_mapPartitions_beyond_partition()
        return "mapPartitions success"
    except Exception as err:
        print ("mapPartitions occur err,{0}".format(err))


def test_mapPartition():
    table = eggroll.parallelize([1, 2, 3, 4, 5], partition=2)
    def f(iterator):
        sum = 0
        for k, v in iterator:
            sum += v
            return sum
    for i in table.mapPartitions(f).collect():
        print (i)


def test_mapPartitions_beyond_partition():
    data = range(5)
    table1 = eggroll.parallelize(data, partition=5)
    table2 = eggroll.parallelize(data, partition=10)
    table1_result = table1.mapPartitions(add).collect()
    table2_result = table2.mapPartitions(add).collect()
    sum1 = 0
    sum2 = 0
    for i in table1_result:
        sum1 += i[1]
    for j in table2_result:
        sum2 += j[1]
    print(sum1, sum(data))
    print(sum2, sum(data))
    # print(list(table1_result), [(4, 5), (3, 7), (1, 2), (0, 1)])
    # print(list(table2_result), [(4, 5), (0, 1), (3, 7), (1, 2)])


def test_mapPartitions_small_partition():
    data = range(5)
    table = eggroll.parallelize(data, partition=2)
    table_result = table.mapPartitions(add).collect()
    sum1 = 0
    for i in table_result:
        sum1 += i[1]
    print(sum1, sum(data))
    # print(list(table_result), [(4,12), (1,3)])


def test_mapPartitions_one_partition():
    data = range(5)
    table = eggroll.parallelize(data, partition=1)
    table_result = table.mapPartitions(add).collect()
    sum1 = 0
    for i in table_result:
        sum1 += i[1]
    print(sum1, sum(data))
    # print(list(table_result), [(4, 15)])


def test_join():
    try:
        test_join_simple()
        test_join_global_param()
        test_join_big_int_key()
        #test_join_big_str_key()
        test_join_easy()
        return "join success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_join_simple():
    data1 = [("a", 1), ("b", 4)]
    data2 = [("a", 2), ("c", 3)]
    table1 = eggroll.parallelize(data1, include_key=True, partition=1)
    table2 = eggroll.parallelize(data2, include_key=True, partition=1)
    value1 = list(table1.join(table2, lambda v1, v2: v1 + v2).collect())
    print(value1, [("a", 3)])


def test_join_global_param():
    data1 = [("a", 1), ("b", 4)]
    data2 = [("a", 2), ("c", 3)]
    table1 = eggroll.parallelize(data1, include_key=True, partition=1)
    table2 = eggroll.parallelize(data2, include_key=True, partition=1)
    n = 5
    value1 = list(table1.join(table2, lambda v1, v2: (v1 + v2) / n).collect())
    print(value1, [("a", 0.6)])
    n = 2
    value2 = list(table1.join(table2, lambda v1, v2: (v1 + v2) / n).collect())
    print(value2, [("a", 1.5)])


def test_join_big_int_key():
    # join: the key is int
    table1_int_key = eggroll.parallelize(range(10), partition=1)
    table2_int_key = eggroll.parallelize(range(5, 15), partition=1)
    intersection_int = table1_int_key.join(table2_int_key, lambda v1, v2: v2)
    length_collect = len(list(intersection_int.collect()))
    length_count = intersection_int.count()
    print(length_collect)
    print(length_collect, length_count)


def test_join_big_str_key():
    # join: the key is string
    list1 = []
    for i, j in enumerate(range(10)):
        list1.append(('a' + str(i), j))
    list2 = []
    for i, j in enumerate(range(5, 15)):
        list2.append(('a' + str(i), j))
    table1_str_key = eggroll.parallelize(list1, include_key=True, partition=2)
    table2_str_key = eggroll.parallelize(list2, include_key=True, partition=2)
    intersection_str = table1_str_key.join(table2_str_key, lambda v1, v2: v2)
    list_intersection = list(intersection_str.collect())
    length_collect = len(list_intersection)
    length_count = intersection_str.count()
    print(length_count)
    print(length_collect, length_count)


def test_join_easy():
    x = eggroll.parallelize([("a", 1), ("b", 4)], include_key=True)
    y = eggroll.parallelize([("a", 2), ("c", 3)], include_key=True)
    print(x.join(y, lambda v1, v2: v1 + v2).collect())


def test_sample():
    try:
        table = eggroll.parallelize(range(100), partition=1)
        value = table.sample(0.1, 81).count()
        print (value)
        return "sample success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_glom():
    try:
        print(eggroll.parallelize([0, 2, 3, 4, 6], partition=1).glom().collect())
        return "glom success"
    except Exception as err:
        print ("occur err,{0}".format(err))


def test_add():
    print(eggroll.parallelize([1, 2, 3, 4, 5]).reduce(add))
    return "add success"

if __name__ == "__main__":
	eggroll.init()
	print ("=========start test eggroll=========")
	test_eggroll_init()
	#test_eggroll_shutdown()
	test_eggroll_table()
	print ("=========end test eggroll=========")


	print ("..................................")
	print ("=========start storeage eggroll=========")
	print (test_save_as())
	print (test_count())
	print (test_collect())
	print (test_put())
	print (test_put_if_absent())
	print (test_put_all())
	print (test_get())
	print (test_destroy())
	print (test_range())
	print (test_delete())
	print ("=========end storeage eggroll=========")

	print ("..................................")
	print ("=========start computation eggroll=========")

	print (test_map())
	print (test_mapValues())
	print (test_reduce())
	print (test_mapPartitions())
	print (test_join())
	print (test_sample())
	print (test_glom())
	print (test_add())
