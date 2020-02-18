#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
import io
import os
import zipfile


def zip_dir_as_bytes(path):
    """
    Zip input directory, return as bytes.
    :param path: input directory.
    :return: 
    """
    with io.BytesIO() as io_bytes:
        with zipfile.ZipFile(io_bytes, 'w', zipfile.ZIP_DEFLATED) as zipper:
            for root, dirs, files in os.walk(path, topdown=False):
                for name in files:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
                for name in dirs:
                    full_path = os.path.join(root, name)
                    relative_path = os.path.relpath(full_path, path)
                    zipper.write(filename=full_path, arcname=relative_path)
        zip_bytes = io_bytes.getvalue()
    return zip_bytes
