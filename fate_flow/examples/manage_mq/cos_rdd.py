import pika, os, pyspark
import json

num_slice = 4
counter = 4 
result = []

credentials = pika.PlainCredentials('federate-test', '123456')
connection = pika.BlockingConnection(pika.ConnectionParameters("localhost", 5672, "federated", credentials))

channel = connection.channel()

sc = pyspark.SparkContext(appName="spark_redis_consumer")
union_rdd = sc.emptyRDD()

for method, properties, body in channel.consume(queue="testing-queue"):
    if(counter == 0):
        break
    else:
        rdd_test = sc.parallelize(json.loads(body), 1)
        union_rdd = union_rdd.union(rdd_test)
        # print("This is message from consumer: ", union_rdd.collect())
        counter = counter - 1

print(union_rdd.countByKey())