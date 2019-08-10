import Mock from 'mockjs'
import { param2Obj } from '../src/utils'

const channelListData = Mock.mock({
  'data|200': [{
    code: '@id',
    channel: '@name',
    channelId: '@integer(100,5000)',
    subChannel: '@name',
    subChannelId: '@integer(100,5000)',
    createTime: '@datetime'
  }]
})

const list = channelListData.data

export default [
  {
    url: '/channel/list',
    type: 'get',
    response: config => {
      const params = param2Obj(config.url)
      if (params.channel) {
      }
      return {
        code: 20000,
        data: list
      }
    }
  },
  {
    url: '/channel/add',
    type: 'post',
    response: config => {
      const params = JSON.parse(config.body)
      params.code = Mock.mock('@id')
      params.createTime = Mock.mock('@date')
      list.unshift(params)
      return {
        code: 20000,
        data: 'success'
      }
    }
  },
  {
    url: '/channel/update',
    type: 'post',
    response: config => {
      const params = JSON.parse(config.body)
      const { code, channel, channelId, subChannel, subChannelId } = params
      for (let i = 0; i < list.length; i++) {
        if (list[i].code === code) {
          list[i].channel = channel
          list[i].channelId = channelId
          list[i].subChannel = subChannel
          list[i].subChannelId = subChannelId
          break
        }
      }
      return {
        code: 20000,
        data: 'success'
      }
    }
  },
  {
    url: '/channel/delete',
    type: 'post',
    response:
      config => {
        const params = param2Obj(config.url)
        for (let i = 0; i < list.length; i++) {
          if (list[i].code === params.code) {
            list.splice(i, 1)
            break
          }
        }
        return {
          code: 20000,
          data: 'success'
        }
      }
  }
]
