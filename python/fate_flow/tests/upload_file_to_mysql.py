import argparse

import pymysql

from fate_arch import storage

database_config = {
    'user': 'root',
    'passwd': 'fate_dev',
    'host': '127.0.0.1',
    'port': 3306
}


class MysqldbHelper(object):
    def __init__(self, host='', user='', passwd='', port='',  database=''):
        self.host = host
        self.user = user
        self.password = passwd
        self.database = database
        self.port = port
        self.con = None
        self.cur = None
        try:
            print(host, user, passwd, port, database)
            self.con = pymysql.connect(host=self.host, user=self.user, passwd=self.password, port=self.port, db=self.database)
            self.cur = self.con.cursor()
        except:
            print("DataBase connect error,please check the db config.")

    def execute(self, sql):
        self.cur.execute(sql)
        results = self.cur.fetchall()


def create_db(namespace):
    conn = pymysql.connect(host=database_config.get('host'),
                           port=database_config.get('port'),
                           user=database_config.get('user'),
                           password=database_config.get('passwd'))
    cursor = conn.cursor()
    cursor.execute("create database if not exists {}".format(namespace))
    print('create db {} success'.format(namespace))
    cursor.close()


def list_to_str(input_list):
    return ','.join(list(map(str, input_list)))


def write_to_db(conf, table_name, file_name, namespace, partitions, head):
    db = MysqldbHelper(**conf)
    table_meta = storage.StorageTableMeta(name=table_name, namespace=namespace)
    create_table = 'create table {}(id varchar(50) NOT NULL, features LONGTEXT, PRIMARY KEY(id))'.format(table_name)
    db.execute(create_table.format(table_name))
    print('create table {}'.format(table_name))

    with open(file_name, 'r') as f:
        if head:
            data_head = f.readline()
            header_source_item = data_head.split(',')
            table_meta.update_metas(schema={'header': ','.join(header_source_item[1:]).strip(), 'sid': header_source_item[0]})
        n = 0
        count = 0
        while True:
            data = list()
            lines = f.readlines(12400)

            if lines:
                sql = 'REPLACE INTO {}(id, features)  VALUES'.format(table_name)
                for line in lines:
                    count += 1
                    values = line.replace("\n", "").replace("\t", ",").split(",")
                    data.append((values[0], list_to_str(values[1:])))
                    sql += '("{}", "{}"),'.format(values[0], list_to_str(values[1:]))
                sql = ','.join(sql.split(',')[:-1]) + ';'
                if n == 0:
                    table_meta.update_metas(part_of_data=data, partitions=partitions)
                n +=1
                db.execute(sql)
                db.con.commit()
            else:
                break
        table_meta.update_metas(count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--namespace', required=True, type=str, help="namespace")
    parser.add_argument('-t', '--table_name', required=True, type=str, help="table_name")
    parser.add_argument('-f', '--file_name', required=True, type=str, help="file_name")
    parser.add_argument('-p', '--partitions', required=True, type=int, help="partitions")
    parser.add_argument('-head', '--head', required=True, type=int, help="head")

    args = parser.parse_args()
    namespace = args.namespace
    table_name = args.table_name
    file_name = args.file_name
    partitions = args.partitions
    head = args.head
    create_db(namespace)
    database_config['database'] = namespace
    write_to_db(database_config, table_name, file_name, namespace, partitions=partitions, head=head)