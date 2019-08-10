import axios from 'axios'
// import store from '@/store'
import { Message } from 'element-ui'
// import store from '@/store'
// import { getToken } from '@/utils/auth'

// axios.defaults.headers.common['Authorization'] = getToken()
// create an axios instance
// console.log(window.location.origin)
const service = axios.create({
  baseURL: window.location.origin,
  withCredentials: false,
  timeout: 10000 // request timeout
})
// request interceptor
service.interceptors.request.use(
  config => {
    // Do something before request is sent
    // if (store.getters.token) {
    //   config.headers['Authorization'] = getToken()
    // }
    return config
  },
  error => {
    // Do something with request error
    console.log(error) // for debug
    Promise.reject(error)
  }
)

// response interceptor
service.interceptors.response.use(
  /**
   * If you want to get information such as headers or status
   * Please return  response => response
   */
  response => {
    const res = response.data
    if (res.code === 0) {
      return new Promise(resolve => {
        // if (store.getters.isOpenReqSimulate) {
        //   setTimeout(function() {
        //     resolve(res)
        //     // console.log('util.request: response:', res)
        //   }, 1000)
        // } else {
        resolve(res)
        // console.log('util.request: response:', res)
        // }
      })
    } else if (res.code === 100) {
      Message({
        message: res.message || res.msg || res.retmsg,
        type: 'warning',
        duration: 3 * 1000
      })
      // return new Promise(resolve => {
      //   // if (store.getters.isOpenReqSimulate) {
      //   //   setTimeout(function() {
      //   //     resolve(res)
      //   //     // console.log('util.request: response:', res)
      //   //   }, 1000)
      //   // } else {
      //   resolve(res)
      //   // console.log('util.request: response:', res)
      //   // }
      // })
      return Promise.reject('warning')
    } else {
      Message({
        message: res.message || res.msg,
        type: 'error',
        duration: 3 * 1000
      })
      return Promise.reject('error')
    }
  },
  error => {
    console.log('err' + error) // for debug
    Message({
      message: error.message,
      type: 'error',
      duration: 5 * 1000
    })
    return Promise.reject(error)
  }
)

export default service
