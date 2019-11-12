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
import datetime
import time

from arch.api import session
from arch.api.utils.core import current_timestamp
from arch.api.utils import log_utils, version_control
from arch.api.utils.dtable_utils import get_table_info, gen_party_version
from federatedml.statistic.intersect.rsa_cache.db_models import DB, IdLibraryCacheInfo, init_database_tables
from federatedml.statistic.intersect.rsa_cache.redis_adaptor import RedisAdaptor

LOGGER = log_utils.getLogger()

'''
################################################################################
########## interface ###########################################################
################################################################################
'''

'''
return: a dictionary contains table_name and namespace, 
'''
def host_get_current_verison(host_party_id, id_type, encrypt_type, tag, timeout=600):
    return get_current_version(id_type, encrypt_type, tag, host_party_id, timeout=timeout)


'''
return: a dictionary contains table_name and namespace, 
'''
def guest_get_current_version(host_party_id, guest_party_id, id_type, encrypt_type, tag, timeout=600):
    return get_current_version(id_type, encrypt_type, tag, host_party_id, guest_party_id=guest_party_id, timeout=timeout)


'''
return: a dictionary contains rsa_n, rsa_e, and rsa_d
'''
def get_rsa_of_current_version(host_party_id, id_type, encrypt_type, tag, timeout=60):
    table_info = host_get_current_verison(host_party_id, id_type, encrypt_type, tag, timeout=timeout)
    if table_info is None:
        LOGGER.info('no cache exists.')
        return None
    namespace = table_info.get('namespace')
    version = table_info.get('table_name')
    if namespace is None or version is None:
        LOGGER.info('host_get_current_verison return None, partyid={}, id_type={}, encrypt_type={}, tag={}.'.format(host_party_id, \
            id_type, encrypt_type, tag))
        return None

    init_database_tables()
    with DB.connection_context():
        LOGGER.info('query cache info, partyid={}, id_type={}, encrypt_type={}, namespace={}, version={}, tag={}'.format(host_party_id, \
            id_type, encrypt_type, namespace, version, tag))
        infos = IdLibraryCacheInfo.select().where(IdLibraryCacheInfo.f_party_id == host_party_id, \
            IdLibraryCacheInfo.f_id_type == id_type, IdLibraryCacheInfo.f_encrypt_type == encrypt_type, \
            IdLibraryCacheInfo.f_tag == tag, IdLibraryCacheInfo.f_namespcae == namespace, IdLibraryCacheInfo.f_version == version)
        if infos:
            info = infos[0]
            rsa_key = {'rsa_n': info.f_rsa_key_n, 'rsa_e': info.f_rsa_key_e, 'rsa_d': info.f_rsa_key_d}
            LOGGER.debug(rsa_key)
            return rsa_key
        else:
            LOGGER.info('query cache info return nil, partyid={}, id_type={}, encrypt_type={}, namespace={}, version={}, tag={}'.format( \
                host_party_id, id_type, encrypt_type, namespace, version, tag))
            return None


def store_cache(dtable, guest_party_id, host_party_id, version, id_type, encrypt_type, tag='Za', namespace=None):
    if namespace is None:
        namespace = gen_cache_namespace(id_type, encrypt_type, tag, host_party_id, guest_party_id=guest_party_id)
    table_config = {}
    table_config['gen_table_info'] = True
    table_config['namespace'] = namespace
    table_config['table_name'] = version
    LOGGER.info(table_config)
    version, namespace = get_table_info(config=table_config, create=True)
    return save_data(dtable, namespace, version)
    

def store_rsa(host_party_id, id_type, encrypt_type, tag, namespace, version, rsa):
    init_database_tables()
    with DB.connection_context():
        LOGGER.info('store rsa and out table info, partyid={}, id_type={}, encrypt_type={}, namespace={}, version={}.'.format(host_party_id, \
            id_type, encrypt_type, namespace, version))
        infos = IdLibraryCacheInfo.select().where(IdLibraryCacheInfo.f_party_id == host_party_id, \
            IdLibraryCacheInfo.f_id_type == id_type, IdLibraryCacheInfo.f_encrypt_type == encrypt_type, \
            IdLibraryCacheInfo.f_tag == tag, IdLibraryCacheInfo.f_namespcae == namespace, IdLibraryCacheInfo.f_version == version)
        is_insert = True
        if infos:
            info = infos[0]
            is_insert = False
        else:
            info = IdLibraryCacheInfo()

        info.f_party_id = host_party_id
        info.f_id_type = id_type
        info.f_encrypt_type = encrypt_type
        info.f_namespcae = namespace
        info.f_version = version
        info.f_tag = tag
        info.f_rsa_key_n = str(rsa.get('rsa_n'))
        info.f_rsa_key_d = str(rsa.get('rsa_d'))
        info.f_rsa_key_e = str(rsa.get('rsa_e'))
        info.f_create_time = current_timestamp()
        if is_insert:
            info.save(force_insert=True)
        else:
            info.save()


def clean_all_cache(host_party_id, id_type, encrypt_type, tag='Za', guest_party_id=None):
    namespace = gen_cache_namespace(id_type, encrypt_type, tag, host_party_id, guest_party_id=guest_party_id)
    session.cleanup(name='*', namespace=namespace, persistent=True)
    version_table = version_control.get_version_table(data_table_namespace=namespace)
    version_table.destroy()


def clean_cache(namespace, version):
    session.cleanup(name=version, namespace=namespace, persistent=True)
    version_control.delete_version(version, namespace)


def clean_all_rsa(host_party_id, id_type, encrypt_type, tag='Za'):
    init_database_tables()
    with DB.connection_context():
        LOGGER.info('clean rsa and out table info, partyid={}, id_type={}, encrypt_type={}, tag={}.'.format(host_party_id, \
            id_type, encrypt_type, tag))
        IdLibraryCacheInfo.delete().where(IdLibraryCacheInfo.f_party_id == host_party_id, \
            IdLibraryCacheInfo.f_id_type == id_type, IdLibraryCacheInfo.f_encrypt_type == encrypt_type, \
            IdLibraryCacheInfo.f_tag == tag).execute()


def clean_rsa(namespace, version):
    init_database_tables()
    with DB.connection_context():
        LOGGER.info('clean rsa and out table info, namespace={}, version={}.'.format(namespace, version))
        IdLibraryCacheInfo.delete().where(IdLibraryCacheInfo.f_namespcae == namespace, \
            IdLibraryCacheInfo.f_version == version).execute()



'''
################################################################################
########## intra-face ###########################################################
################################################################################
'''
def gen_cache_namespace(id_type, encrypt_type, tag, host_party_id, guest_party_id=None, data_type='id_library_cache'):
    if guest_party_id is None:
        guest_party_id = 'all'
    return '#'.join([data_type, str(host_party_id), str(guest_party_id), id_type, encrypt_type, tag])


def get_current_version(id_type, encrypt_type, tag, host_party_id, guest_party_id=None, timeout=600):
    config = {}
    config['gen_table_info'] = True
    config['namespace'] = gen_cache_namespace(id_type, encrypt_type, tag, host_party_id, guest_party_id=guest_party_id)

    LOGGER.info(config)
    table_info = get_table_info_without_create(table_config=config)
    LOGGER.info(table_info)
    
    if table_info.get('table_name'):
        LOGGER.info('table exists, namepsace={}, version={}.'.format(table_info.get('namespace'), table_info.get('table_name')))
        return table_info
    
    redis_adapter = RedisAdaptor()
    cache_job = redis_adapter.get(config['namespace'])
    if cache_job is None:
        LOGGER.info('neighter table, nor cache job exists, namepsace={}.'.format(config['namespace']))
        if guest_party_id:
            redis_adapter.setex(config['namespace'], 'guest_get_current_version')
        return None
    
    for i in range(timeout):
        cache_job = redis_adapter.get(config['namespace'])
        if cache_job is None:
            table_info = get_table_info_without_create(table_config=config)
            if table_info.get('table_name'):
                LOGGER.info('after cache job finish, get table info, namepsace={}, version={}.'.format(table_info.get('namespace'), table_info.get('table_name')))
                return table_info
            else:
                LOGGER.info('after cache job finish, table not exist, namepsace={}.'.format(config['namespace']))
                return None
        time.sleep(1)
    LOGGER.info('wait cache job timeout, get version fail, namepsace={}.'.format(config['namespace']))
    return None


def save_data(data_inst, namespace, version):
    redis_adapter = RedisAdaptor()
    redis_adapter.setex(namespace, version)

    persistent_table = data_inst.save_as(namespace=namespace, name=version)
    LOGGER.info("save data to namespace={}, name={}".format(persistent_table._namespace, persistent_table._name))

    session.save_data_table_meta(
        {'schema': data_inst.schema, 'header': data_inst.schema.get('header', [])},
        data_table_namespace=persistent_table._namespace, data_table_name=persistent_table._name)

    version_log = "[AUTO] save data at %s." % datetime.datetime.now()
    version_control.save_version(name=persistent_table._name, namespace=persistent_table._namespace, version_log=version_log)

    redis_adapter.delete(namespace)

    LOGGER.info('save table done, namepsace={}, version={}.'.format(persistent_table._namespace, persistent_table._name))
    return {'table_name': persistent_table._name, 'namespace': persistent_table._namespace}


def get_table_info_without_create(table_config):
    table_name, namespace = get_table_info(config=table_config, create=False)
    return {'table_name': table_name, 'namespace': namespace}

def gen_cache_version(namespace, create=False):
    return gen_party_version(namespace=namespace, create=create)




