from peewee import CharField, BigIntegerField, IntegerField, CompositeKey

from arch.api.utils.core_utils import current_timestamp
from fate_arch.common.log import getLogger
from fate_flow.db.db_models import DataBaseModel, DB

LOGGER = getLogger()

DB = DB


class TableMeta(DataBaseModel):
    f_table_name = CharField(max_length=500)
    f_table_namespace = CharField(max_length=500)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_records = IntegerField(null=True, default=0)
    f_partitions = IntegerField(null=True, default=0)

    class Meta:
        db_table = "t_table_meta"
        primary_key = CompositeKey('f_table_namespace', 'f_table_name')


def _update_table_meta(namespace, name, records):
    try:
        with DB.connection_context():
            metas = TableMeta.select().where(TableMeta.f_table_namespace == namespace,
                                             TableMeta.f_table_name == name)
            is_insert = True
            if metas:
                meta = metas[0]
                is_insert = False
            else:
                meta = TableMeta()
                meta.f_table_namespace = namespace
                meta.f_table_name = name
                meta.f_create_time = current_timestamp()
                meta.f_records = 0
            meta.f_records = meta.f_records + records
            meta.f_update_time = current_timestamp()
            if is_insert:
                meta.save(force_insert=True)
            else:
                meta.save()
    except Exception as e:
        LOGGER.error("update_table_meta exception:{}.".format(e))


def _get_table_meta(namespace, name):
    try:
        with DB.connection_context():
            metas = TableMeta.select().where(TableMeta.f_table_namespace == namespace,
                                             TableMeta.f_table_name == name)
            if metas:
                return metas[0]
    except Exception as e:
        LOGGER.error("update_table_meta exception:{}.".format(e))


def _delete_table_meta(namespace, name):
    try:
        with DB.connection_context():
            TableMeta.delete().where(TableMeta.f_table_namespace == namespace,
                                     TableMeta.f_table_name == name).execute()
    except Exception as e:
        LOGGER.error("delete_table_meta {}, {}, exception:{}.".format(namespace, name, e))
