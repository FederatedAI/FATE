import Mock from 'mockjs'
import { param2Obj } from '../src/utils'

const Random = Mock.Random
const list = []
let pid = 0

for (let i = 1; i <= 5; i++) {
  list.push({
    pid: ++pid,
    name: `project${i}`,
    datasets: Random.integer(10, 50),
    experiments: Random.integer(10, 50),
    jobs: Random.integer(10, 50),
    type: Random.integer(1, 2),
    desc: Random.string(10, 200),
    time: Random.datetime('yyyy-MM-dd HH:mm:ss')
  })
}

export default [
  {
    url: '/project/query',
    type: 'get',
    response: config => {
      // const params = param2Obj(config.url)
      // console.log(params)
      return {
        code: 0,
        data: list
      }
    }
  },
  {
    url: '/project/add',
    type: 'post',
    response: config => {
      const params = JSON.parse(config.body)
      params.pid = ++pid
      params.time = Mock.mock('@date')
      params.datasets = Random.integer(10, 50)
      params.experiments = Random.integer(10, 50)
      params.jobs = Random.integer(10, 50)
      list.unshift(params)
      return {
        code: 0,
        message: 'success',
        data: { pid: params.pid }
      }
    }
  },
  {
    url: '/project/update',
    type: 'post',
    response: config => {
      const params = JSON.parse(config.body)
      const { pid, type, desc, name } = params
      for (let i = 0; i < list.length; i++) {
        if (list[i].pid === pid) {
          list[i].name = name
          list[i].desc = desc
          list[i].type = type
          break
        }
      }
      return {
        code: 0,
        data: 'success'
      }
    }
  },
  {
    url: '/project/delete',
    type: 'post',
    response:
      config => {
        const pid = JSON.parse(config.body).pid
        list.map((item, index) => {
          if (item.pid === pid) {
            list.splice(index, 1)
          }
        })
        return {
          code: 0,
          data: 'success'
        }
      }
  }
]
