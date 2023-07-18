#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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
from typing import List, Union

import pandas as pd
from fate.arch import Context
from fate.arch.dataframe import PandasReader
from fate.components.core import GUEST, HOST, Role, cpn, params


@cpn.component(roles=[GUEST, HOST])
def toy_example(
        ctx: Context,
        role: Role,
        output_data: cpn.dataframe_output(roles=[GUEST, HOST]),
        json_model_output: cpn.json_model_output(roles=[GUEST, HOST]),
        data_num: cpn.parameter(type=params.conint(gt=1), desc="data_num", optional=False),
        partition: cpn.parameter(type=params.conint(gt=1), desc="data_partition", optional=False),
):
    pd_df = pd.DataFrame([[str(i), str(i), i] for i in range(data_num)], columns=["sample_id", "match_id", "x0"])
    reader = PandasReader(sample_id_name="sample_id", match_id_name="match_id", dtype="float64", partition=partition)
    df = reader.to_frame(ctx, pd_df)

    if role == "guest":
        ctx.hosts.put("guest_index", df.get_indexer(target="sample_id"))
        host_indexes = ctx.hosts[0].get("host_index")
        final_df = df.loc(host_indexes, preserve_order=True)
    else:
        guest_indexes = ctx.guest.get("guest_index")
        final_df = df.loc(guest_indexes)
        ctx.guest.put("host_index", final_df.get_indexer(target="sample_id"))

    assert final_df.shape[0] == data_num, f"data num should be {data_num} instead of {final_df}"

    output_data.write(final_df)

    json_model_output.write({"test_role": role})
