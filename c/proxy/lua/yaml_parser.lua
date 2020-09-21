-- YAMLParserLite = class("YAMLParserLite")

-- function YAMLParserLite:initialize()
-- end

-- function YAMLParserLite:parse(yaml)
--     local lines = {}
--     for line in string.gmatch(yaml..'\n', '(.-)\n') do
--         table.insert(lines, line)
--     end

--     local docs = parse_documents(lines)
--     if #docs == 1 then
--         return docs[1]
--     end
--     return docs
-- end

-- 以上是为了 配合个人已有结构

local schar = string.char
local ssub, gsub = string.sub, string.gsub
local sfind, smatch = string.find, string.match
local tinsert, tremove = table.insert, table.remove

local UNESCAPES = {
    ['0'] = "\x00", z = "\x00", N    = "\x85",
    a = "\x07",     b = "\x08", t    = "\x09",
    n = "\x0a",     v = "\x0b", f    = "\x0c",
    r = "\x0d",     e = "\x1b", ['\\'] = '\\',
}

-- help function
local function select(list, pred)
  local selected = {}
  for i = 0, #list do
    local v = list[i]
    if v and pred(v, i) then
      tinsert(selected, v)
    end
  end
  return selected
end

-- return: indent_count, left_string
local function count_indent(line)
  local _, j = sfind(line, '^%s+')
  if not j then
    return 0, line
  end
  return j, ssub(line, j+1)
end

local function trim(str)
    return string.gsub(str, "^%s*(.-)%s*$", "%1")
end

local function ltrim(str)
  return smatch(str, "^%s*(.-)$")
end

local function rtrim(str)
  return smatch(str, "^(.-)%s*$")
end

local function isemptyline(line)
    return line == '' or sfind(line, '^%s*$') or sfind(line, '^%s*#')
end

local function startswith(haystack, needle)
    return ssub(haystack, 1, #needle) == needle
end

local function startswithline(line, needle)
    return startswith(line, needle) and isemptyline(ssub(line, #needle+1))
end

-- class
local class = {__meta={}}
function class.__meta.__call(cls, ...)
  local self = setmetatable({}, cls)
  if cls.__init then
    cls.__init(self, ...)
  end
  return self
end

function class.def(base, type, cls)
  base = base or class
  local mt = {__metatable=base, __index=base}
  for k, v in pairs(base.__meta) do mt[k] = v end
  cls = setmetatable(cls or {}, mt)
  cls.__index = cls
  cls.__metatable = cls
  cls.__type = type
  cls.__meta = mt
  return cls
end

local types = {
  null = class:def('null'),
  map = class:def('map'),
  seq = class:def('seq'),
}

local Null = types.null
function Null.__tostring() return 'yaml.null' end
function Null.isnull(v)
  if v == nil then return true end
  if type(v) == 'table' and getmetatable(v) == Null then return true end
  return false
end
local null = Null()

-- implement function
local function parse_string(line, stopper)

    stopper = stopper or ''
    local q = ssub(line, 1, 1)
    if q == ' ' or q == '\t' then
      return parse_string(ssub(line, 2))
    end

    if q == "'" then
      local i = sfind(line, "'", 2, true)
      if not i then
        return nil, line
      end
      return ssub(line, 2, i-1), ssub(line, i+1)
    end

    if q == '"' then
      local i, buf = 2, ''
      while i < #line do
        local c = ssub(line, i, i)
        if c == '\\' then
          local n = ssub(line, i+1, i+1)
          if UNESCAPES[n] ~= nil then
            buf = buf..UNESCAPES[n]
          elseif n == 'x' then
            local h = ssub(i+2,i+3)
            if sfind(h, '^[0-9a-fA-F]$') then
              buf = buf..schar(tonumber(h, 16))
              i = i + 2
            else
              buf = buf..'x'
            end
          else
            buf = buf..n
          end
          i = i + 1
        elseif c == q then
          break
        else
          buf = buf..c
        end
        i = i + 1
      end
      return buf, ssub(line, i+1)
    end

    if q == '-' or q == ':' then
      if ssub(line, 2, 2) == ' ' or #line == 1 then
        return nil, line
      end
    end

    local buf = ''
    while #line > 0 do
      local c = ssub(line, 1, 1)
      if sfind(stopper, c, 1, true) then
        break
      elseif c == ':' and (ssub(line, 2, 2) == ' ' or #line == 1) then
        break
      elseif c == '#' and (ssub(buf, #buf, #buf) == ' ') then
        break
      else
        buf = buf..c
      end
      line = ssub(line, 2)
    end
    return rtrim(buf), line
end

local function parse_flowstyle(line, lines)
  local stack = {}
  while true do
    if #line == 0 then
      if #lines == 0 then
        break
      else
        line = tremove(lines, 1)
      end
    end
    local c = ssub(line, 1, 1)
    if c == '#' then
      line = ''
    elseif c == ' ' or c == '\t' or c == '\r' or c == '\n' then
      line = ssub(line, 2)
    elseif c == '{' or c == '[' then
      tinsert(stack, {v={},t=c})
      line = ssub(line, 2)
    elseif c == ':' then
      local s = tremove(stack)
      tinsert(stack, {v=s.v, t=':'})
      line = ssub(line, 2)
    elseif c == ',' then
      local value = tremove(stack)
      if value.t == ':' or value.t == '{' or value.t == '[' then error() end
      if stack[#stack].t == ':' then
        -- map
        local key = tremove(stack)
        stack[#stack].v[key.v] = value.v
      elseif stack[#stack].t == '{' then
        -- set
        stack[#stack].v[value.v] = true
      elseif stack[#stack].t == '[' then
        -- seq
        tinsert(stack[#stack].v, value.v)
      end
      line = ssub(line, 2)
    elseif c == '}' then
      if stack[#stack].t == '{' then
        if #stack == 1 then break end
        stack[#stack].t = '}'
        line = ssub(line, 2)
      else
        line = ','..line
      end
    elseif c == ']' then
      if stack[#stack].t == '[' then
        if #stack == 1 then break end
        stack[#stack].t = ']'
        line = ssub(line, 2)
      else
        line = ','..line
      end
    else
      local s, rest = parse_string(line, ',{}[]')
      if not s then
        error('invalid flowstyle line: '..line)
      end
      tinsert(stack, {v=s, t='s'})
      line = rest
    end
  end
  return stack[1].v, line
end

local function parse_scalar(line, lines)

    line = ltrim(line)
    line = gsub(line, '%s*#.*$', '')
  
    if line == '' or line == '~' then
      return null
    end
    
    if startswith(line, '{') or startswith(line, '[') then
      return parse_flowstyle(line, lines)
    end

    local s, _ = parse_string(line)
    if s and s ~= line then
      return s
    end

    -- Special cases
    if sfind('\'"!$', ssub(line, 1, 1), 1, true) then
      error('unsupported line: '..line)
    end
  
    if startswithline(line, '{}') then
      return {}
    end
    if startswithline(line, '[]') then
      return {}
    end
  
    -- Regular unquoted string
    local v = line
    if v == 'null' or v == 'Null' or v == 'NULL'then
      return null
    elseif v == 'true' or v == 'True' or v == 'TRUE' then
      return true
    elseif v == 'false' or v == 'False' or v == 'FALSE' then
      return false
    elseif v == '.inf' or v == '.Inf' or v == '.INF' then
      return math.huge
    elseif v == '+.inf' or v == '+.Inf' or v == '+.INF' then
      return math.huge
    elseif v == '-.inf' or v == '-.Inf' or v == '-.INF' then
      return -math.huge
    elseif v == '.nan' or v == '.NaN' or v == '.NAN' then
      return 0 / 0
    elseif sfind(v, '^[%+%-]?[0-9]+$') or sfind(v, '^[%+%-]?[0-9]+%.$')then
      return tonumber(v)
    elseif sfind(v, '^[%+%-]?[0-9]+%.[0-9]+$') then
      return tonumber(v)
    end
    return v
end

local parse_map

local function parse_seq(line, lines, indent)
  local seq = setmetatable({}, types.seq)
  if line ~= '' then
    error()
  end
  while #lines > 0 do
    line = lines[1]

    local level = count_indent(line)
    if level < indent and indent ~= -1 then
      return seq
    elseif level > indent and indent ~= -1 then
      error("found bad indenting in line: ".. line)
    end

    local i, j = sfind(line, '%-%s+')
    if not i then
      i, j = sfind(line, '%-$')
      if not i then
        return seq
      end
    end
    local rest = ssub(line, j+1)

    if sfind(rest, '^[^\'\"%s]*:') then
      local indent2 = j
      lines[1] = string.rep(' ', indent2)..rest
      tinsert(seq, parse_map('', lines, indent2))
    elseif isemptyline(rest) then
      tremove(lines, 1)
      if #lines == 0 then
        tinsert(seq, null)
        return seq
      end
      if sfind(lines[1], '^%s*%-') then
        local nextline = lines[1]
        local indent2 = count_indent(nextline)
        if indent2 == indent then
          tinsert(seq, null)
        else
          tinsert(seq, parse_seq('', lines, indent2))
        end
      else
        local nextline = lines[1]
        local indent2 = count_indent(nextline)
        tinsert(seq, parse_map('', lines, indent2))
      end
    elseif rest then
      tremove(lines, 1)
      local tmp = parse_scalar(rest, lines)
      tinsert(seq, tmp)
    end
  end
  return seq
end

function parse_map(line, lines, indent)
  if not isemptyline(line) then
    error('not map line: '..line)
  end
  local map = setmetatable({}, types.map)
  while #lines > 0 do
    line = lines[1]
  
    local level, _ = count_indent(line)
    if level < indent then
      return map
    elseif level > indent then
      error("found bad indenting in line: ".. line)
    end

    local key
    local s, rest = parse_string(line)
    if s and startswith(rest, ':') then
      local sc = parse_scalar(s, {})
      if sc and type(sc) ~= 'string' then
        key = sc
      else
        key = s
      end
      line = ssub(rest, 2)
    else
      error("failed to classify line: "..line)
    end

    if map[key] ~= nil then
      print("found a duplicate key '"..key.."' in line: "..line)
      local suffix = 1
      while map[key..'__'..suffix] do
        suffix = suffix + 1
      end
      key = key ..'_'..suffix
    end

    line = ltrim(line)

    if not isemptyline(line) then
      tremove(lines, 1)
      line = ltrim(line)
      map[key] = parse_scalar(line, lines)
    else
      tremove(lines, 1)
      if #lines == 0 then
        map[key] = null
        return map;
      end
      if sfind(lines[1], '^%s*%-') then
        local indent2 = count_indent(lines[1])
        map[key] = parse_seq('', lines, indent2)
      else
        local indent2 = count_indent(lines[1])
        if indent >= indent2 then
          map[key] = null
        else
          map[key] = parse_map('', lines, indent2)
        end
      end
    end
  end
  return map
end

local function parse_documents(lines)
    lines = select(lines, function(s) return not isemptyline(s) end)

    if #lines == 1 and not sfind(lines[1], '^%s*%-') then
      local line = lines[1]
      line = ltrim(line)
      return parse_scalar(line, lines)
    end

    local root = {}
    while #lines > 0 do
        local line = lines[1]
        if sfind(line, '^%s*%-') then
            tinsert(root, parse_seq('', lines, -1))
        elseif sfind(line, '^%s*[^%s]') then
            local level = count_indent(line)
            tinsert(root, parse_map('', lines, level))
        else
            error('parse error: '..line)
        end
    end

    if #root > 1 and Null.isnull(root[1]) then
        tremove(root, 1)
        return root
    end

    return root

end

local function parse(yaml)
    local lines = {}
    for line in string.gmatch(yaml..'\n', '(.-)\n') do
        table.insert(lines, line)
    end

    local docs = parse_documents(lines)
    if #docs == 1 then
        return docs[1]
    end
    return docs
end

return {
  null = null,
  parse = parse,
}

