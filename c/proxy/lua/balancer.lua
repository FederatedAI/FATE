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
local ngx = ngx
local ngx_balancer = require "ngx.balancer"

local function balance()
    local fate_cluster_server = ngx.ctx.fate_cluster_server
    local ok, err = ngx_balancer.set_current_peer(fate_cluster_server)
    if not ok then
        utils.exit_abnormally('failed to set current peer: ' .. err, ngx.HTTP_SERVICE_UNAVAILABLE)
    end
end

balance()

