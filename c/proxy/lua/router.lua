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
local math = require "math"
local string = require "string"

local function get_server_address(server)
    return string.format("%s:%s", server["host"], server["port"])
end

local function get_dest_server(dest_party_id, dest_service)
    ngx.log(ngx.INFO, string.format("try to get %s %s server", dest_party_id, dest_service))

    local route = route_table.get_route()
    local party_services = route:get(dest_party_id)
    local server
    if party_services ~= nil then
        local service = party_services[dest_service]
        server = get_server_address(service[math.random(1, #service)])
    else
        local default_proxy = route:get("default")["proxy"]
        server = get_server_address(default_proxy[math.random(1, #default_proxy)])
    end
    ngx.log(ngx.INFO, string.format("get %s %s server: %s", dest_party_id, dest_service, server))
    return server
end

local function get_request_dest()
    local headers = ngx.req.get_headers()
    return headers
end

function routing()
    local request_headers = get_request_dest()
    local dest_forward_server = get_dest_server(tonumber(request_headers["dest-party-id"]), request_headers["service"])
    ngx.ctx.fate_cluster_server = dest_forward_server
end

routing()

