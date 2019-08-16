/**
 * Created by jiachenpan on 16/11/18.
 */

export function parseTime(time, cFormat) {
  if (arguments.length === 0) {
    return null
  }
  const format = cFormat || '{y}-{m}-{d} {h}:{i}:{s}'
  let date
  if (typeof time === 'object') {
    date = time
  } else {
    if ((typeof time === 'string') && (/^[0-9]+$/.test(time))) {
      time = parseInt(time)
    }
    if ((typeof time === 'number') && (time.toString().length === 10)) {
      time = time * 1000
    }
    date = new Date(time)
  }
  const formatObj = {
    y: date.getFullYear(),
    m: date.getMonth() + 1,
    d: date.getDate(),
    h: date.getHours(),
    i: date.getMinutes(),
    s: date.getSeconds(),
    a: date.getDay()
  }
  const time_str = format.replace(/{(y|m|d|h|i|s|a)+}/g, (result, key) => {
    let value = formatObj[key]
    // Note: getDay() returns 0 on Sunday
    if (key === 'a') {
      return [][value]
    }
    if (result.length > 0 && value < 10) {
      value = '0' + value
    }
    return value || 0
  })
  return time_str
}

export function trimId(str) {
  const strArr = str.substr(1).split('')
  let l = 0
  for (let i = 0; i < strArr.length; i++) {
    if (strArr[i].charCodeAt(0) === 0) {
      l = i
      break
    }
  }
  return str.substr(1, l)
}

export function formatSeconds(seconds, milisecond = true) {
  if (milisecond) {
    seconds /= 1000
  }
  let h = Math.floor(seconds / 3600)
  let m = Math.floor((seconds / 60 % 60))
  let s = Math.floor((seconds % 60))
  const formatNum = function(num) {
    let str = ''
    if (num < 1) {
      str = '00'
    } else if (num < 10) {
      str = `0${num}`
    } else {
      str = num.toString()
    }
    return str
  }
  h = formatNum(h)
  m = formatNum(m)
  s = formatNum(s)
  return `${h}:${m}:${s}`
}

export function filterLineArr(arr) {
  const obj = {}
  arr.forEach(p => {
    const x = p[0]
    if (obj[x] === undefined || obj[x] < x) {
      obj[x] = p
    }
  })

  return Object.values(obj)
}

export function initWebSocket(url, onopen, onmessage, onclose = null) {
  const baseUrl = window.location.origin
  const baseWsUrl = baseUrl.replace(/http|https/g, 'ws')
  const instance = new WebSocket(baseWsUrl + url)
  instance.onopen = onopen
  instance.onmessage = onmessage
  instance.onerror = () => {
    try {
      this.initWebSocket(url, onopen, onmessage, onclose = null)
    } catch (e) {
      console.log('websoket error:', e)
    }
  }
  instance.onclose = function() {
  }
  return instance
}

export function formatTime(time, option) {
  if (('' + time).length === 10) {
    time = parseInt(time) * 1000
  } else {
    time = +time
  }
  const d = new Date(time)
  const now = Date.now()

  const diff = (now - d) / 1000

  if (diff < 30) {
    return ''
  } else if (diff < 3600) {
    // less 1 hour
    return Math.ceil(diff / 60) + ''
  } else if (diff < 3600 * 24) {
    return Math.ceil(diff / 3600) + ''
  } else if (diff < 3600 * 24 * 2) {
    return ''
  }
  if (option) {
    return parseTime(time, option)
  } else {
    return (
      d.getMonth() +
      1 +
      '' +
      d.getDate() +
      '' +
      d.getHours() +
      '' +
      d.getMinutes() +
      ''
    )
  }
}

export function getQueryObject(url) {
  url = url == null ? window.location.href : url
  const search = url.substring(url.lastIndexOf('?') + 1)
  const obj = {}
  const reg = /([^?&=]+)=([^?&=]*)/g
  search.replace(reg, (rs, $1, $2) => {
    const name = decodeURIComponent($1)
    let val = decodeURIComponent($2)
    val = String(val)
    obj[name] = val
    return rs
  })
  return obj
}

export function jsonToTableHeader(obj, label) {
  const header = [{ prop: 'name', label }]
  let flag = true
  const data = []
  for (const key in obj) {
    const content = obj[key]
    if (flag) {
      const arr = Object.keys(content)
      arr.forEach(item => {
        header.push({
          prop: item,
          label: item
        })
      })
      flag = false
    }
    const row = Object.assign({ name: key }, content)
    data.push(row)
  }
  return { header, data }
}

/**
 * @param {Sting} input value
 * @returns {number} output value
 */
export function byteLength(str) {
  // returns the byte length of an utf8 string
  let s = str.length
  for (let i = str.length - 1; i >= 0; i--) {
    const code = str.charCodeAt(i)
    if (code > 0x7f && code <= 0x7ff) {
      s++
    } else if (code > 0x7ff && code <= 0xffff) s += 2
    if (code >= 0xDC00 && code <= 0xDFFF) i--
  }
  return s
}

export function cleanArray(actual) {
  const newArray = []
  for (let i = 0; i < actual.length; i++) {
    if (actual[i]) {
      newArray.push(actual[i])
    }
  }
  return newArray
}

export function param(json) {
  if (!json) return ''
  return cleanArray(
    Object.keys(json).map(key => {
      if (json[key] === undefined) return ''
      return encodeURIComponent(key) + '=' + encodeURIComponent(json[key])
    })
  ).join('&')
}

export function param2Obj(url) {
  const search = url.split('?')[1]
  if (!search) {
    return {}
  }
  return JSON.parse(
    '{"' +
    decodeURIComponent(search)
      .replace(/"/g, '\\"')
      .replace(/&/g, '","')
      .replace(/=/g, '":"')
      .replace(/\+/g, ' ') +
    '"}'
  )
}

export function html2Text(val) {
  const div = document.createElement('div')
  div.innerHTML = val
  return div.textContent || div.innerText
}

export function objectMerge(target, source) {
  /* Merges two  objects,
     giving the last one precedence */

  if (typeof target !== 'object') {
    target = {}
  }
  if (Array.isArray(source)) {
    return source.slice()
  }
  Object.keys(source).forEach(property => {
    const sourceProperty = source[property]
    if (typeof sourceProperty === 'object') {
      target[property] = objectMerge(target[property], sourceProperty)
    } else {
      target[property] = sourceProperty
    }
  })
  return target
}

export function toggleClass(element, className) {
  if (!element || !className) {
    return
  }
  let classString = element.className
  const nameIndex = classString.indexOf(className)
  if (nameIndex === -1) {
    classString += '' + className
  } else {
    classString =
      classString.substr(0, nameIndex) +
      classString.substr(nameIndex + className.length)
  }
  element.className = classString
}

export const pickerOptions = [
  {
    text: 'today',
    onClick(picker) {
      const end = new Date()
      const start = new Date(new Date().toDateString())
      end.setTime(start.getTime())
      picker.$emit('pick', [start, end])
    }
  },
  {
    text: '',
    onClick(picker) {
      const end = new Date(new Date().toDateString())
      const start = new Date()
      start.setTime(end.getTime() - 3600 * 1000 * 24 * 7)
      picker.$emit('pick', [start, end])
    }
  },
  {
    text: '',
    onClick(picker) {
      const end = new Date(new Date().toDateString())
      const start = new Date()
      start.setTime(start.getTime() - 3600 * 1000 * 24 * 30)
      picker.$emit('pick', [start, end])
    }
  },
  {
    text: '',
    onClick(picker) {
      const end = new Date(new Date().toDateString())
      const start = new Date()
      start.setTime(start.getTime() - 3600 * 1000 * 24 * 90)
      picker.$emit('pick', [start, end])
    }
  }
]

export function getTime(type) {
  if (type === 'start') {
    return new Date().getTime() - 3600 * 1000 * 24 * 90
  } else {
    return new Date(new Date().toDateString())
  }
}

export function debounce(func, wait, immediate) {
  let timeout, args, context, timestamp, result

  const later = function() {
    // 据上一次触发时间间隔
    const last = +new Date() - timestamp

    // 上次被包装函数被调用时间间隔 last 小于设定时间间隔 wait
    if (last < wait && last > 0) {
      timeout = setTimeout(later, wait - last)
    } else {
      timeout = null
      if (!immediate) {
        result = func.apply(context, args)
        if (!timeout) context = args = null
      }
    }
  }

  return function(...args) {
    context = this
    timestamp = +new Date()
    const callNow = immediate && !timeout
    if (!timeout) timeout = setTimeout(later, wait)
    if (callNow) {
      result = func.apply(context, args)
      context = args = null
    }

    return result
  }
}

/**
 * This is just a simple version of deep copy
 * Has a lot of edge cases bug
 * If you want to use a perfect deep copy, use lodash's _.cloneDeep
 */
export function deepClone(source) {
  if (!source && typeof source !== 'object') {
    throw new Error('error arguments', 'deepClone')
  }
  const targetObj = source.constructor === Array ? [] : {}
  Object.keys(source).forEach(keys => {
    if (source[keys] && typeof source[keys] === 'object') {
      targetObj[keys] = deepClone(source[keys])
    } else {
      targetObj[keys] = source[keys]
    }
  })
  return targetObj
}

export function deepCloneArr(arr) {
  const newArr = []
  arr.forEach(item => {
    newArr.push(item)
  })
  return newArr
}

export function random(min, max) {
  if (min > max) {
    min = [max, max = min][0]
  }
  const range = max - min
  return (min + Math.round(Math.random() * range))
}

export function simpleDeepClone(source) {
  return JSON.parse(JSON.stringify(source))
}

export function uniqueArr(arr) {
  return Array.from(new Set(arr))
}

export function createUniqueString() {
  const timestamp = +new Date() + ''
  const randomNum = parseInt((1 + Math.random()) * 65536) + ''
  return (+(randomNum + timestamp)).toString(32)
}

export function hasClass(ele, cls) {
  return !!ele.className.match(new RegExp('(\\s|^)' + cls + '(\\s|$)'))
}

export function addClass(ele, cls) {
  if (!hasClass(ele, cls)) ele.className += ' ' + cls
}

export function removeClass(ele, cls) {
  if (hasClass(ele, cls)) {
    const reg = new RegExp('(\\s|^)' + cls + '(\\s|$)')
    ele.className = ele.className.replace(reg, ' ')
  }
}

/**
 * excel表格导出
 */
export function exportExcel(header, data, filename = 'text-export', autoWidth = true, bookType = 'xlsx') {
  // console.log(header, data)
  import('./vendor/Export2Excel').then(excel => {
    excel.export_json_to_excel({
      // 表头 必填 例如['Id', 'Title', 'Author', 'Readings', 'Date']
      header,
      // 具体数据 必填 例如[[1,'标题','作者','内容','20190416'],[2,'标题','作者','内容','20190416']]
      data,
      // 非必填
      filename,
      // 非必填
      autoWidth,
      // 非必填
      bookType
    })
  })
}
