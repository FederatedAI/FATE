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
local new_timer = ngx.timer.at
local _M = {
    _VERSION = '0.1'
}

-- alternatively: local lrucache = require "resty.lrucache.pureffi"
local lrucache = require "resty.lrucache"

-- we need to initialize the cache on the lua module level so that
-- it can be shared by all the requests served by each nginx worker process:
local route_cache, err = lrucache.new(500)  -- allow up to 500 items in the cache
if not route_cache then
    error("failed to create the cache: " .. (err or "unknown"))
end

local function reload_route_table()
    route_cache:set("9999", { fateflow = "127.0.0.1:9360" })
    route_cache:set("10000", { fateflow = "127.0.0.1:9362" })
    ngx.log(ngx.INFO, "reload route table")
end

local function reload()
    reload_route_table()
    local ok, err = new_timer(5, reload)
    if not ok then
        if err ~= "process exiting" then
            errlog("failed to create timer: ", err)
        end
        reload_route_table()
        return
    end
end

function _M.get_route()
    return route_cache
end

function _M.start()
    reload()
end

return _M
