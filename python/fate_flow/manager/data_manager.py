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
from fate_flow.db.db_models import DB, TrackingMetric, DataTableTrackingModel


class DataTableTracker(object):
    @classmethod
    @DB.connection_context()
    def create_table_tracker(cls, table_name, table_namespace, entity_info):
        tracker = DataTableTrackingModel()
        tracker.f_table_name = table_name
        tracker.f_table_namespace = table_namespace
        for k, v in entity_info.items():
            attr_name = 'f_%s' % k
            if hasattr(DataTableTrackingModel, attr_name):
                setattr(tracker, attr_name, v)
        if entity_info.get("have_parent"):
            parent_trackers = DataTableTrackingModel.select().where(
                DataTableTrackingModel.f_table_name == entity_info.get("parent_table_name"),
                DataTableTrackingModel.f_table_namespace == entity_info.get("parent_table_namespace")).order_by(DataTableTrackingModel.f_create_time.desc())
            if not parent_trackers:
                raise Exception(f"table {table_name} {table_namespace} no found parent")
            parent_tracker = parent_trackers[0]
            if parent_tracker.f_have_parent:
                tracker.f_source_table_name = parent_tracker.f_source_table_name
                tracker.f_source_table_namespace = parent_tracker.f_source_table_namespace
            else:
                tracker.f_source_table_name = parent_tracker.f_table_name
                tracker.f_source_table_namespace = parent_tracker.f_table_namespace
        rows = tracker.save(force_insert=True)
        if rows != 1:
            raise Exception("Create {} failed".format(tracker))
        return tracker

    @classmethod
    @DB.connection_context()
    def get_parent_table(cls, table_name, table_namespace):
        filters = [operator.attrgetter('f_table_name')(DataTableTrackingModel) == table_name,
                   operator.attrgetter('f_table_namespace')(DataTableTrackingModel) == table_namespace]
        trackers = DataTableTrackingModel.select().where(*filters)
        if trackers:
            tracker = trackers[0]
            if tracker.f_have_parent:
                parent_table_info = {"parent_table_name": tracker.f_parent_table_name,
                                     "parent_table_namespace": tracker.f_parent_table_namespace,
                                     "source_table_name": trackers.f_source_table_name,
                                     "source_table_namespace": tracker.f_table_namespace,
                                     "have_parent": True}
            else:
                parent_table_info = {"have_parent": False}
        else:
            raise Exception(f"no found table: table name {table_name}, table namespace {table_namespace}")
        return parent_table_info


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
