########################################################
# Copyright 2019-2020 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import random, string

def RandomString(stringLength=6):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def RandomNumberString(stringLength=6):
    letters = string.octdigits
    return ''.join(random.choice(letters) for i in range(stringLength))