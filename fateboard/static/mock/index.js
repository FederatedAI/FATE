import Mock from 'mockjs'
import mocks from './mocks'
import store from '../src/store'

// Fix an issue with setting withCredentials = true, cross-domain request lost cookies
// https://github.com/nuysoft/Mock/issues/300
Mock.XHR.prototype.proxy_send = Mock.XHR.prototype.send
Mock.XHR.prototype.send = function() {
  if (this.custom.xhr) {
    this.custom.xhr.withCredentials = this.withCredentials || false
  }
  this.proxy_send(...arguments)
}
// Mock.setup({
//   timeout: '350-600'
// })

// User
for (let i = 0; i < mocks.length; i++) {
  const item = mocks[i]
  Mock.mock(new RegExp(item.url), item.type, item.response)
}

// Mock.mock(/\/user\/login/, 'post', userAPI.login)
// Mock.mock(/\/user\/info/, 'get', userAPI.getInfo)
// Mock.mock(/\/user\/logout/, 'post', userAPI.logout)

// // Table
// Mock.mock(/\/table\/list/, 'get', tableAPI.list)
// if (store.getters.isOpenReqSimulate) {
//   Mock.setup({ timeout: 1000 })
// }
export default Mock
