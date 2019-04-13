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
#
from peewee_migrate import Router
from playhouse.pool import PooledMySQLDatabase
from arch.task_manager.settings import DATABASE
data_base_config = DATABASE.copy()
# TODO: create instance according to the engine
engine = data_base_config.pop("engine")
db_name = data_base_config.pop("name")
DB = PooledMySQLDatabase(db_name, **data_base_config)

router = Router(DB)

# Create migration
router.create('task_manager')

# Run migration/migrations
router.run('task_manager')

# Run all unapplied migrations
router.run()
