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


def underline_to_pascal(var):
    ws = ['%s%s' % (w[0].upper(), w[1:]) for w in var.split('_')]
    return ''.join(ws)


def pascal_to_underline(var):
    ws = []
    last_i = 0
    for i in range(1, len(var), 1):
        if 65 <= ord(var[i]) <= 90:
            ws.append(var[last_i:i].lower())
            last_i = i
    ws.append(var[last_i:].lower())
    return '_'.join(ws)


def underline_to_camel(var):
    ws = underline_to_pascal(var)
    ws = '%s%s' %(ws[0].lower(), ws[1:])
    return ws


def camel_to_pascal(var):
    ws = '%s%s' % (var[0].upper(), var[1:])
    return ws


def list_feature_to_fate_str(input_list):
    str1 = ''
    size = len(input_list)
    for i in range(size):
        if i == size - 1:
            str1 += str(input_list[i])
        else:
            str1 += str(input_list[i]) + ','
    return str1
