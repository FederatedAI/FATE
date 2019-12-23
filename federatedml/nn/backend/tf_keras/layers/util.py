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


from tensorflow.python.keras import initializers


def _get_initializer(initializer, seed):
    if not seed:
        return initializer

    initializer_class = getattr(initializers, initializer, None)
    if initializer_class:
        initializer_instance = initializer_class()
        if hasattr(initializer_instance, "seed"):
            initializer_instance.seed = seed
        return initializer_instance

    return initializer

