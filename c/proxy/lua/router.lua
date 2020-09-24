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

local function get_upstream_server(dest_party_id, dest_service)
    ngx.log(ngx.INFO, string.format("try to get %s %s upstream", dest_party_id, dest_service))

    local route = route_table.get_route()
    local party_services = route:get(dest_party_id)
    if party_services ~= nil then
        local server = party_services[dest_service]
        ngx.log(ngx.INFO, server)
        return server
    else
        local default_party_services = route:get("default")
    end
    return server
end

local function get_request_dest()
    local headers = ngx.req.get_headers()
    return headers
end

function routing()
    local request_headers = get_request_dest()
    local forward_server = get_upstream_server(request_headers["dest-party-id"], request_headers["service"])
    ngx.ctx.fate_cluster_server = forward_server
end

routing()

