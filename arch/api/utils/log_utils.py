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
from arch.api.utils import file_utils


class LoggerFactory(object):
    TYPE = "FILE"
    LEVEL = logging.DEBUG
    logger_dict = {}
    global_handler_dict = {}

    LOG_DIR = None
    PARENT_LOG_DIR = None

    append_to_parent_log = None

    lock = RLock()
    # CRITICAL = 50
    # FATAL = CRITICAL
    # ERROR = 40
    # WARNING = 30
    # WARN = WARNING
    # INFO = 20
    # DEBUG = 10
    # NOTSET = 0
    levels = (10, 20, 30, 40)

    @staticmethod
    def set_directory(directory=None, parent_log_dir=None, append_to_parent_log=None, force=False):
        if parent_log_dir:
           LoggerFactory.PARENT_LOG_DIR = parent_log_dir
        if append_to_parent_log:
            LoggerFactory.append_to_parent_log = append_to_parent_log
        with LoggerFactory.lock:
            if not directory:
                directory = os.path.join(file_utils.get_project_base_directory(), 'logs')
            if not LoggerFactory.LOG_DIR or force:
                LoggerFactory.LOG_DIR = directory
            os.makedirs(LoggerFactory.LOG_DIR, exist_ok=True)
            for loggerName, ghandler in LoggerFactory.global_handler_dict.items():
                for className, (logger, handler) in LoggerFactory.logger_dict.items():
                    logger.removeHandler(ghandler)
                ghandler.close()
            LoggerFactory.global_handler_dict={}
            for className, (logger, handler) in LoggerFactory.logger_dict.items():
                logger.removeHandler(handler)
                _hanlder=None
                if handler:
                    handler.close()
                if className != "default":
                    _hanlder = LoggerFactory.get_handler(className)
                    logger.addHandler(_hanlder)
                LoggerFactory.assemble_global_handler(logger)
                LoggerFactory.logger_dict[className] = logger, _hanlder

    @staticmethod
    def get_logger(class_name=None):
        with LoggerFactory.lock:
            if class_name in LoggerFactory.logger_dict.keys():
                logger, hanlder = LoggerFactory.logger_dict[class_name]
                if not logger:
                    logger, handler = LoggerFactory.init_logger(class_name)
            else:
                logger, handler = LoggerFactory.init_logger(class_name)
            return logger

    @staticmethod
    def get_global_hanlder(logger_name, level=None, log_dir=None):
        if not LoggerFactory.LOG_DIR:
            return logging.StreamHandler()
        if log_dir:
            logger_name_key = logger_name + "_" + log_dir
        else:
            logger_name_key = logger_name + "_" + LoggerFactory.LOG_DIR
        # if loggerName not in LoggerFactory.globalHandlerDict:
        if logger_name_key not in LoggerFactory.global_handler_dict:
            with LoggerFactory.lock:
                if logger_name_key not in LoggerFactory.global_handler_dict:
                    handler = LoggerFactory.get_handler(logger_name, level, log_dir)
                    LoggerFactory.global_handler_dict[logger_name_key]=handler
        return LoggerFactory.global_handler_dict[logger_name_key]

    @staticmethod
    def get_handler(class_name, level=None, log_dir=None):
        if not LoggerFactory.LOG_DIR or not class_name:
            return logging.StreamHandler()
        formatter = logging.Formatter('"%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"')
        if not log_dir:
            log_file = os.path.join(LoggerFactory.LOG_DIR, "{}.log".format(class_name))
        else:
            log_file = os.path.join(log_dir, "{}.log".format(class_name))
        handler = TimedRotatingFileHandler(log_file,
                                           when='D',
                                           interval=1,
                                           backupCount=14,
                                           delay=True)

        if level:
            handler.level=level

        handler.setFormatter(formatter)
        return handler

    @staticmethod
    def init_logger(class_name):
        with LoggerFactory.lock:
            logger = logging.getLogger(class_name)
            logger.setLevel(LoggerFactory.LEVEL)
            handler = None
            if class_name:
                handler = LoggerFactory.get_handler(class_name)
                logger.addHandler(handler)
                LoggerFactory.logger_dict[class_name] = logger, handler

            else:
                LoggerFactory.logger_dict["default"] = logger, handler

            LoggerFactory.assemble_global_handler(logger)

            return logger, handler

    @staticmethod
    def assemble_global_handler(logger):
        if LoggerFactory.LOG_DIR:
            for level in LoggerFactory.levels:
                if level >= LoggerFactory.LEVEL:
                    level_logger_name = logging._levelToName[level]
                    logger.addHandler(LoggerFactory.get_global_hanlder(level_logger_name, level))
        if LoggerFactory.append_to_parent_log and LoggerFactory.PARENT_LOG_DIR:
            for level in LoggerFactory.levels:
                if level >= LoggerFactory.LEVEL:
                    level_logger_name = logging._levelToName[level]
                    logger.addHandler(LoggerFactory.get_global_hanlder(level_logger_name, level, LoggerFactory.PARENT_LOG_DIR))


def setDirectory(directory=None):
    LoggerFactory.set_directory(directory)


def setLevel(level):
    LoggerFactory.LEVEL = level


def getLogger(className=None, useLevelFile=False):
    if className is None:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        className = 'stat'
    return LoggerFactory.get_logger(className)
