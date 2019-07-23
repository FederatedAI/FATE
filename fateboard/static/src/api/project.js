import request from '@/utils/request'

export function getProjectList(params) {
  return request({
    url: '/project/query',
    method: 'get',
    params
  })
}

export function deleteProject(pid) {
  return request({
    url: '/project/delete',
    method: 'post',
    data: pid
  })
}

export function updateProject(data) {
  return request({
    url: '/project/update',
    method: 'post',
    data
  })
}

export function addProject(data) {
  return request({
    url: '/project/add',
    method: 'post',
    data
  })
}
