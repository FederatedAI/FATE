import request from '@/utils/request'

export function getAllSSHConfig() {
  return request({
    url: '/ssh/all',
    method: 'get',
    params: {}
  })
}
export function getAllSSHStatus() {
  return request({
    url: '/ssh/checkStatus',
    method: 'get',
    params: {}
  })
}
export function addSSHConfig(data) {
  return request({
    url: '/ssh/ssh',
    method: 'post',
    data
  })
}

export function removeSSHConfig(ip) {
  return request({
    url: `/ssh/ssh`,
    method: 'delete',
    data: { ip }
  })
}
