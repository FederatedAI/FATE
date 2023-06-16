#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from fate.interface import Address


class EggRollAddress(Address):
    def __init__(self, home=None, name=None, namespace=None):
        self.name = name
        self.namespace = namespace
        self.home = home

    def __hash__(self):
        return (self.home, self.name, self.namespace).__hash__()

    def __str__(self):
        return f"EggRollAddress(name={self.name}, namespace={self.namespace})"

    def __repr__(self):
        return self.__str__()


class HDFSAddress(Address):
    def __init__(self, name_node=None, path=None):
        self.name_node = name_node
        self.path = path

    def __hash__(self):
        return (self.name_node, self.path).__hash__()

    def __str__(self):
        return f"HDFSAddress(name_node={self.name_node}, path={self.path})"

    def __repr__(self):
        return self.__str__()


class HiveAddress(Address):
    def __init__(
        self,
        host=None,
        name=None,
        port=10000,
        username=None,
        database="default",
        auth_mechanism="PLAIN",
        password=None,
    ):
        self.host = host
        self.username = username
        self.port = port
        self.database = database
        self.auth_mechanism = auth_mechanism
        self.password = password
        self.name = name

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"HiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class LinkisHiveAddress(Address):
    def __init__(
        self,
        host="127.0.0.1",
        port=9001,
        username="",
        database="",
        name="",
        run_type="hql",
        execute_application_name="hive",
        source={},
        params={},
    ):
        self.host = host
        self.port = port
        self.username = username
        self.database = database if database else f"{username}_ind"
        self.name = name
        self.run_type = run_type
        self.execute_application_name = execute_application_name
        self.source = source
        self.params = params

    def __hash__(self):
        return (self.host, self.port, self.database, self.name).__hash__()

    def __str__(self):
        return f"LinkisHiveAddress(database={self.database}, name={self.name})"

    def __repr__(self):
        return self.__str__()


class LocalFSAddress(Address):
    def __init__(self, path):
        self.path = path

    def __hash__(self):
        return (self.path).__hash__()

    def __str__(self):
        return f"LocalFSAddress(path={self.path})"

    def __repr__(self):
        return self.__str__()
