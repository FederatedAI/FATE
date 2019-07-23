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

export function killJob(jobId) {
  return request({
    url: '/job/v1/pipeline/job/stop',
    method: 'post',
    data: { 'job_id': jobId }
  })
}

export function getJobDetails(jobId) {
  return request({
    url: `/job/query/${jobId}`,
    method: 'get'
  })
}

export function getDAGDpencencies(jobId) {
  return request({
    url: '/v1/pipeline/dag/dependencies',
    method: 'post',
    data: { job_id: jobId }
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

export function queryLog({ componentId, jobId, begin, end, type = 'default' }) {
  return request({
    url: `/queryLogWithSize/${componentId}/${jobId}/${type}/${begin}/${end}  `,
    method: 'get'
  })
}

