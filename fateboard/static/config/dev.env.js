'use strict'
const merge = require('webpack-merge')
const prodEnv = require('./prod.env')

// module.exports = merge(prodEnv, {
//   NODE_ENV: '"development"',
//   BASE_API: '"http://172.16.153.113:8080"',
//   WEBSOCKET_BASE_API: '"ws://172.16.153.113:8080"'
// })
module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  BASE_API: '"http://172.16.153.113:9090"',
  WEBSOCKET_BASE_API: '"ws://172.16.153.113:9090"'
})
