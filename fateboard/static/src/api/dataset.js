import request from '@/utils/request'

export function getProjectList(params) {
  return request({
    url: '/project/query',
    method: 'get',
    params
  })
}

export function addChannel(data) {
  return request({
    url: '/channel/add',
    method: 'post',
    data
  })
}

export function updateChannel(data) {
  return request({
    url: '/channel/update',
    method: 'post',
    data
  })
}

export function deleteChannel(data) {
  return request({
    url: '/channel/delete',
    method: 'post',
    data
  })
}
