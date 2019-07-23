import request from '@/utils/request'

// 获取图表类型数据
export function getMetrics(data) {
  return request({
    url: '/v1/tracking/component/metrics',
    method: 'post',
    data
  })
}
export function getMetricData(data) {
  return request({
    url: '/v1/tracking/component/metric_data',
    method: 'post',
    data
  })
}
export function getDataOutput(data) {
  return request({
    url: '/v1/tracking/component/output/data',
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

