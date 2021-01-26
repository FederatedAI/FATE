--
--  Copyright 2019 The FATE Authors. All Rights Reserved.
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.
--
local _M = {
    _VERSION = '0.1'
}

local ngx = ngx
local route_table = require "route_table"

local function routing()
    local request_headers = ngx.req.get_headers()
    local dest_env = request_headers["dest-party-id"]
    if dest_env == nil then
        dest_env = request_headers["dest-env"]
    end
    ngx.ctx.dest_cluster = route_table.get_dest_server(dest_env, request_headers["service"])
end

routing()

