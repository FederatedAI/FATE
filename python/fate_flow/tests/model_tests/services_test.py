import os
import time
import unittest
from unittest.mock import patch

from kazoo.client import KazooClient
from kazoo.exceptions import NodeExistsError, NoNodeError

from fate_flow.db import db_services
from fate_flow.errors.error_services import *
from fate_flow.db.db_models import DB, MachineLearningModelInfo as MLModel
from fate_flow import settings


model_download_url = 'http://127.0.0.1:9380/v1/model/transfer/arbiter-10000_guest-9999_host-10000_model/202105060929263278441'
escaped_model_download_url = '/FATE-SERVICES/flow/online/transfer/providers/http%3A%2F%2F127.0.0.1%3A9380%2Fv1%2Fmodel%2Ftransfer%2Farbiter-10000_guest-9999_host-10000_model%2F202105060929263278441'


class TestZooKeeperDB(unittest.TestCase):

    def setUp(self):
        # required environment: ZOOKEEPER_HOSTS
        # optional environment: ZOOKEEPER_USERNAME, ZOOKEEPER_PASSWORD
        config = {
            'hosts': os.environ['ZOOKEEPER_HOSTS'].split(','),
            'use_acl': False,
        }
        username = os.environ.get('ZOOKEEPER_USERNAME')
        password = os.environ.get('ZOOKEEPER_PASSWORD')
        if username and password:
            config.update({
                'use_acl': True,
                'username': username,
                'password': password,
            })

        with patch.object(db_services.ServiceRegistry, 'USE_REGISTRY', 'ZooKeeper'), \
                patch.object(db_services.ServiceRegistry, 'ZOOKEEPER', config):
            self.service_db = db_services.service_db()

    def test_services_db(self):
        self.assertEqual(type(self.service_db), db_services.ZooKeeperDB)
        self.assertNotEqual(type(self.service_db), db_services.FallbackDB)
        self.assertEqual(type(self.service_db.client), KazooClient)

    def test_zookeeper_not_configured(self):
        with patch.object(db_services.ServiceRegistry, 'USE_REGISTRY', True), \
            patch.object(db_services.ServiceRegistry, 'ZOOKEEPER', {'hosts': None}), \
                self.assertRaisesRegex(ZooKeeperNotConfigured, ZooKeeperNotConfigured.message):
            db_services.service_db()

    def test_missing_zookeeper_username_or_password(self):
        with patch.object(db_services.ServiceRegistry, 'USE_REGISTRY', True), \
            patch.object(db_services.ServiceRegistry, 'ZOOKEEPER', {
                'hosts': ['127.0.0.1:2281'],
                'use_acl': True,
            }), self.assertRaisesRegex(
                MissingZooKeeperUsernameOrPassword, MissingZooKeeperUsernameOrPassword.message):
            db_services.service_db()

    def test_get_znode_path(self):
        self.assertEqual(self.service_db._get_znode_path('fateflow', model_download_url), escaped_model_download_url)

    def test_crud(self):
        self.service_db._insert('fateflow', model_download_url)
        self.assertIn(model_download_url, self.service_db.get_urls('fateflow'))

        self.service_db._delete('fateflow', model_download_url)
        self.assertNotIn(model_download_url, self.service_db.get_urls('fateflow'))

    def test_insert_exists_node(self):
        self.service_db._delete('servings', 'http://foo/bar')
        self.service_db._insert('servings', 'http://foo/bar')

        with self.assertRaises(NodeExistsError):
            self.service_db.client.create(self.service_db._get_znode_path('servings', 'http://foo/bar'), makepath=True)

        self.service_db._insert('servings', 'http://foo/bar')
        self.service_db._delete('servings', 'http://foo/bar')

    def test_delete_not_exists_node(self):
        self.service_db._delete('servings', 'http://foo/bar')

        with self.assertRaises(NoNodeError):
            self.service_db.client.delete(self.service_db._get_znode_path('servings', 'http://foo/bar'))

        self.service_db._delete('servings', 'http://foo/bar')

    def test_connection_closed(self):
        self.service_db._insert('fateflow', model_download_url)
        self.assertIn(model_download_url, self.service_db.get_urls('fateflow'))

        self.service_db.client.stop()
        self.service_db.client.start()
        self.assertNotIn(model_download_url, self.service_db.get_urls('fateflow'))

    def test_register_models(self):
        try:
            os.remove(DB.database)
        except FileNotFoundError:
            pass

        MLModel.create_table()
        for x in range(1, 101):
            job_id = str(time.time())
            model = MLModel(
                f_role='host', f_party_id='100', f_job_id=job_id,
                f_model_id=f'foobar#{x}', f_model_version=job_id,
                f_initiator_role='host', f_work_mode=0
            )
            model.save(force_insert=True)
        self.assertEqual(db_services.models_group_by_party_model_id_and_model_version().count(), 100)

        with patch.object(self.service_db, '_insert') as insert:
            self.service_db.register_models()
        self.assertEqual(insert.call_count, 100)
        with patch.object(self.service_db, '_delete') as delete:
            self.service_db.unregister_models()
        self.assertEqual(delete.call_count, 100)

        os.remove(DB.database)


class TestFallbackDB(unittest.TestCase):

    def setUp(self):
        with patch.object(db_services.ServiceRegistry, 'USE_REGISTRY', False):
            self.service_db = db_services.service_db()

    def test_get_urls(self):
        self.assertEqual(self.service_db._get_urls('fateflow'), ['http://127.0.0.1:9380/v1/model/transfer'])
        self.assertEqual(self.service_db._get_urls('servings'), ['http://127.0.0.1:8000'])

    def test_crud(self):
        self.service_db._insert('fateflow', model_download_url)
        self.assertNotIn(model_download_url, self.service_db.get_urls('fateflow'))

        self.service_db._delete('fateflow', model_download_url)
        self.assertNotIn(model_download_url, self.service_db.get_urls('fateflow'))

    def test_get_model_download_url(self):
        self.assertEqual(db_services.get_model_download_url('foo-111#bar-222', '20210616'),
                         'http://127.0.0.1:9380/v1/model/transfer/foo-111_bar-222/20210616')

    def test_not_supported_service(self):
        with self.assertRaisesRegex(ServiceNotSupported, 'The service foobar is not supported'):
            self.service_db.get_urls('foobar')


if __name__ == '__main__':
    unittest.main()
