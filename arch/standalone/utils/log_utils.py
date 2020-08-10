#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from arch.standalone.utils import file_utils
import inspect
from threading import RLock


class LoggerFactory(object):
    TYPE = "FILE"
    LEVEL = logging.DEBUG
    loggerDict = {}
    LOG_DIR = None
    lock = RLock()

    @staticmethod
    def setDirectory(directory=None):
        with LoggerFactory.lock:
            if not directory:
                directory = os.path.join(file_utils.get_project_base_directory(), 'logs')
            LoggerFactory.LOG_DIR = directory
            os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)
            for className, (logger, handler) in LoggerFactory.loggerDict.items():
                logger.removeHandler(handler)
                handler.close()
                _hanlder = LoggerFactory.get_hanlder(className)
                logger.addHandler(_hanlder)
                LoggerFactory.loggerDict[className] = logger, _hanlder

    @staticmethod
    def getLogger(className):
        with LoggerFactory.lock:
            if className in LoggerFactory.loggerDict.keys():
                logger, hanlder = LoggerFactory.loggerDict[className]
                if not logger:
                    logger, handler = LoggerFactory.__initLogger(className)
            else:
                logger, handler = LoggerFactory.__initLogger(className)
            return logger

    @staticmethod
    def get_hanlder(className):
        if not LoggerFactory.LOG_DIR:
            return logging.StreamHandler()
        formatter = logging.Formatter('"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"')
        log_file = os.path.join(LoggerFactory.LOG_DIR, "{}.log".format(className))
        handler = TimedRotatingFileHandler(log_file,
                                           when='H',
                                           interval=4,
                                           backupCount=7,
                                           delay=True)

        handler.setFormatter(formatter)
        return handler

    @staticmethod
    def __initLogger(className):
        with LoggerFactory.lock:
            logger = logging.getLogger(className)
            logger.setLevel(LoggerFactory.LEVEL)
            handler = LoggerFactory.get_hanlder(className)
            logger.addHandler(handler)

            LoggerFactory.loggerDict[className] = logger, handler
            return logger, handler


def setDirectory(directory=None):
    LoggerFactory.setDirectory(directory)


def setLevel(level):
    LoggerFactory.LEVEL = level


def getLogger(className=None):
    if className is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        className = os.path.splitext(os.path.basename(module.__file__))[0]
    return LoggerFactory.getLogger(className)
