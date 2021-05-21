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
from fate_flow.settings import stat_logger
from fate_flow.db.db_models import DB, TrackingMetric


@DB.connection_context()
def query_data_view(**kwargs):
    filters = []
    for f_n, f_v in kwargs.items():
        attr_name = 'f_%s' % f_n
        if hasattr(DataView, attr_name):
            filters.append(operator.attrgetter('f_%s' % f_n)(DataView) == f_v)
    if filters:
        data_views = DataView.select().where(*filters)
    else:
        data_views = []
    return [data_view for data_view in data_views]


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
    if metric_info.get('model'):
        sql = drop_metric_data_mode(metric_info.get('model'))
    else:
        sql = delete_metric_data_from_db(metric_info)
    return sql


@DB.connection_context()
def drop_metric_data_mode(model):
    try:
        drop_sql = 'drop table t_tracking_metric_{}'.format(model)
        DB.execute_sql(drop_sql)
        stat_logger.info(drop_sql)
        return drop_sql
    except Exception as e:
        stat_logger.exception(e)
        raise e


@DB.connection_context()
def delete_metric_data_from_db(metric_info):
    try:
        job_id = metric_info['job_id']
        metric_info.pop('job_id')
        delete_sql = 'delete from t_tracking_metric_{}  where f_job_id="{}"'.format(job_id[:8], job_id)
        for k, v in metric_info.items():
            if hasattr(TrackingMetric, "f_" + k):
                connect_str = " and f_"
                delete_sql = delete_sql + connect_str + k + '="{}"'.format(v)
        DB.execute_sql(delete_sql)
        stat_logger.info(delete_sql)
        return delete_sql
    except Exception as e:
        stat_logger.exception(e)
        raise e
