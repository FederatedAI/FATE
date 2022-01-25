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
import operator

from fate_arch import storage
from fate_flow.db.db_models import DB, TrackingMetric


def delete_tables_by_table_infos(output_data_table_infos):
    data = []
    status = False
    for output_data_table_info in output_data_table_infos:
        table_name = output_data_table_info.f_table_name
        namespace = output_data_table_info.f_table_namespace
        table_info = {'table_name': table_name, 'namespace': namespace}
        if table_name and namespace and table_info not in data:
            with storage.Session.build(name=table_name, namespace=namespace) as storage_session:
                table = storage_session.get_table()
                try:
                    table.destroy()
                    data.append(table_info)
                    status = True
                except:
                    pass
    return status, data


def delete_metric_data(metric_info):
    status = delete_metric_data_from_db(metric_info)
    return f"delete status: {status}"


@DB.connection_context()
def delete_metric_data_from_db(metric_info):
    tracking_metric_model = type(TrackingMetric.model(table_index=metric_info.get("job_id")[:8]))
    operate = tracking_metric_model.delete().where(*get_delete_filters(tracking_metric_model, metric_info))
    return operate.execute() > 0


def get_delete_filters(tracking_metric_model, metric_info):
    delete_filters = []
    primary_keys = ["job_id", "role", "party_id", "component_name"]
    for key in primary_keys:
        if key in metric_info:
            delete_filters.append(operator.attrgetter("f_%s" % key)(tracking_metric_model) == metric_info[key])
    return delete_filters
