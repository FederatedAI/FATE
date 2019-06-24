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

import logging
from logging.handlers import TimedRotatingFileHandler
import os
import inspect
from threading import RLock
from enum import Enum

from arch.api.utils import file_utils


class LoggerFactory(object):
    TYPE = "FILE"
    LEVEL = logging.DEBUG
    loggerDict = {}
    globalHandlerDict = {}

    LOG_DIR = None
    lock = RLock()
    # CRITICAL = 50
    # FATAL = CRITICAL
    # ERROR = 40
    # WARNING = 30
    # WARN = WARNING
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0

    levels = (10, 20,30, 40)


    @staticmethod
    def setDirectory(directory=None):
        with LoggerFactory.lock:
            if not directory:
                directory = os.path.join(file_utils.get_project_base_directory(), 'logs')
            LoggerFactory.LOG_DIR = directory
            os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)
            for loggerName, handler in LoggerFactory.globalHandlerDict.items():
                handler.close()
            LoggerFactory.globalHandlerDict={}
            for className, (logger, handler) in LoggerFactory.loggerDict.items():
                logger.removeHandler(handler)
                handler.close()
                _hanlder = LoggerFactory.get_hanlder(className)
                logger.addHandler(_hanlder)
                LoggerFactory.assembleGloableHandler(logger)
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
    def get_globle_hanlder(loggerName,level=None):
        if not LoggerFactory.LOG_DIR:
            return logging.StreamHandler()

        if loggerName not in LoggerFactory.globalHandlerDict:
            with LoggerFactory.lock:
                if(loggerName not in LoggerFactory.globalHandlerDict):
                    handler = LoggerFactory.get_hanlder(loggerName,level)
                    LoggerFactory.globalHandlerDict[loggerName]=handler


        return LoggerFactory.globalHandlerDict[loggerName]




    @staticmethod
    def get_hanlder(className,level=None):
        if not LoggerFactory.LOG_DIR:
            return logging.StreamHandler()
        formatter = logging.Formatter('"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"')
        log_file = os.path.join(LoggerFactory.LOG_DIR, "{}.log".format(className))
        handler = TimedRotatingFileHandler(log_file,
                                           when='H',
                                           interval=4,
                                           backupCount=7,
                                           delay=True)

        if(level):
            handler.level=level

        handler.setFormatter(formatter)
        return handler




    @staticmethod
    def __initLogger(className):
        with LoggerFactory.lock:
            logger = logging.getLogger(className)
            logger.setLevel(LoggerFactory.LEVEL)
            handler = LoggerFactory.get_hanlder(className)
           # LoggerFactory.get_handlder_use_level()
            logger.addHandler(handler)
            LoggerFactory.assembleGloableHandler(logger)
            LoggerFactory.loggerDict[className] = logger, handler
            return logger, handler

    @staticmethod
    def assembleGloableHandler(logger):
        if LoggerFactory.LOG_DIR:
            for level in LoggerFactory.levels:
                if level >= LoggerFactory.LEVEL:
                    levelLoggerName = logging._levelToName[level]
                    logger.addHandler(LoggerFactory.get_globle_hanlder(levelLoggerName, level))


def setDirectory(directory=None):
    LoggerFactory.setDirectory(directory)


def setLevel(level):
    LoggerFactory.LEVEL = level


def getLogger(className=None,useLevelFile=False):
    if className is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        className = os.path.splitext(os.path.basename(module.__file__))[0]
    return LoggerFactory.getLogger(className)







