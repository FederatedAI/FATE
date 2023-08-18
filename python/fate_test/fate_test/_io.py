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
import click
import loguru

from fate_test._ascii import HEAD, TAIL, BENCHMARK


# noinspection PyPep8Naming
class echo(object):
    _file = None

    @classmethod
    def set_file(cls, file):
        cls._file = file

    @classmethod
    def echo(cls, message, **kwargs):
        click.secho(message, **kwargs)
        click.secho(message, file=cls._file, **kwargs)

    @classmethod
    def file(cls, message, **kwargs):
        click.secho(message, file=cls._file, **kwargs)

    @classmethod
    def stdout(cls, message, **kwargs):
        click.secho(message, **kwargs)

    @classmethod
    def stdout_newline(cls):
        click.secho("")

    @classmethod
    def welcome(cls, banner_type="testsuite"):
        if banner_type == "testsuite":
            cls.echo(HEAD)
        elif banner_type == "benchmark":
            cls.echo(BENCHMARK)

    @classmethod
    def farewell(cls):
        cls.echo(TAIL)

    @classmethod
    def flush(cls):
        import sys
        sys.stdout.flush()


def set_logger(name):
    loguru.logger.remove()
    loguru.logger.add(name, level='ERROR', delay=True)
    return loguru.logger


LOGGER = loguru.logger
