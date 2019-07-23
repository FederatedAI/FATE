import Vue from 'vue'
// import 'normalize.css/normalize.css' // A modern alternative to CSS resets
import './theme/index.css'
import ElementUI from 'element-ui'
// import 'element-ui/lib/theme-chalk/index.css'
import locale from 'element-ui/lib/locale/lang/zh-CN' // lang i18n

import '@/styles/index.scss' // global css

import App from './App'
import store from './store'
import router from './router'

import '@/icons' // icon

import '@/iconfont/iconfont.css' // iconfont
// import '@/permission' // permission control
/**
 * This project originally used easy-mock to simulate data,
 * but its official service is very unstable,
 * and you can build your own service if you need it.
 * So here I use Mock.js for local emulation,
 * it will intercept your request, so you won't see the request in the network.
 * If you remove `../mock` it will automatically request easy-mock data.
 */

// if (process.env.NODE_ENV === 'development') {
//   require('../mock') // simulation data
// }

Vue.use(ElementUI, { locale })

Vue.config.productionTip = false

Vue.filter('projectTypeFormat', type => {
  return store.getters.projectType[type - 1].label || 'Unknown'
})
new Vue({
  el: '#app',
  router,
  store,
  render: h => h(App)
})
