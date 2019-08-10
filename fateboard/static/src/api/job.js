import request from '@/utils/request'

export function getAllJobs({ total, pno, psize = 10 }) {
  return request({
    url: `/job/query/all/${total}/${pno}/${psize}`,
    method: 'get',
    params: {}
  })
}

export function getJobsTotal() {
  return request({
    url: '/job/query/totalrecord',
    method: 'get',
    params: {}
  })
}

export function getAllJobsStatus(params) {
  return request({
    url: '/job/query/status',
    method: 'get',
    params
  })
}

export function killJob(data) {
  return request({
    url: '/job/v1/pipeline/job/stop',
    method: 'post',
    data
  })
}

export function getJobDetails({ job_id, role, party_id }) {
  return request({
    url: `/job/query/${job_id}/${role}/${party_id}`,
    method: 'get'
  })
}

export function getDAGDpencencies(data) {
  return request({
    url: '/v1/pipeline/dag/dependencies',
    method: 'post',
    data
  })
}

export function getComponentPara(data) {
  return request({
    url: '/v1/tracking/component/parameters',
    method: 'post',
    data
  })
}

export function getModelOutput(data) {
  return request({
    url: '/v1/tracking/component/output/model',
    method: 'post',
    data
  })
}

export function queryLog({ componentId, job_id, role, party_id, begin, end, type }) {
  return request({
    url: `/queryLogWithSize/${job_id}/${role}/${party_id}/${componentId}/${type}/${begin}/${end}`,
    method: 'get'
  })
}

export function queryLogSize({ componentId = 'default', job_id, role, party_id, type }) {
  return request({
    url: `/queryLogSize/${job_id}/${role}/${party_id}/${componentId}/${type}`,
    method: 'get'
  })
}

