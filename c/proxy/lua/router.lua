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
function get_upstream_server(request_headers)
    -- TODO: Gets the destination address from the routing table based on the header information
    local server = "127.0.0.1:9360"
    return server
end

function get_request_dest()
    local headers = ngx.req.get_headers()
    for k, v in pairs(headers) do
        ngx.log(ngx.INFO, k)
        ngx.log(ngx.INFO, v)
    end
    return headers
end

function routing()
    local request_headers = get_request_dest()
    local forward_server = get_upstream_server(request_headers)
    ngx.ctx.fate_cluster_server = forward_server
    -- local ok, err = ngx_balancer.set_current_peer(forward_server)
    -- if not ok then
    --     utils.exit_abnormally('failed to set current peer: ' .. err, ngx.HTTP_SERVICE_UNAVAILABLE)
    -- end
end

routing()

