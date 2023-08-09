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
from .ecdh._run import psi_ecdh


def psi_run(ctx, df, protocol="ecdh_psi", curve_type="curve25519"):
    if protocol == "ecdh_psi":
        return psi_ecdh(ctx, df, curve_type=curve_type)
    else:
        raise ValueError(f"PSI protocol={protocol} does not implemented yet.")
