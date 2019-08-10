<template>
  <div class="dashboard-container bg-dark app-container">

    <h3 class="app-title flex space-between">
      <span>Dashboard</span>
      <!--<p>Job: <span class="text-primary pointer" @click="toDetails()">{{ jobId }}</span></p>-->
      <p>Job: <span>{{ jobId }}</span></p>
    </h3>

    <el-row :gutter="24" class="dash-board-list">
      <el-col :span="8">
        <div v-loading="datasetLoading" class="col dataset-info shadow">

          <h3 class="list-title">DATASET INFO</h3>

          <el-row v-for="(row,index) in datasetList" :key="index" :gutter="4" class="dataset-row">
            <el-col :span="6" :offset="2">
              <div class="dataset-item">
                <p class="name dataset-title">{{ row.role }}</p>
                <p v-if="row.options.length===1" class="value">{{ row.roleValue }}</p>
                <el-select v-else v-model="row.roleValue">
                  <el-option
                    v-for="(option,index) in row.options"
                    :key="index"
                    :value="option.value"
                    :label="option.label"
                  />
                  {{ row.roleValue }}
                </el-select>
              </div>
            </el-col>

            <el-col :span="14">
              <div class="dataset-item">
                <p class="name">DATASET</p>
                <p class="value">
                  <el-tooltip
                    :content="row.datasetData? Object.values(row.datasetData[row.roleValue]).join(', ') : ''"
                    placement="top">
                    <span>{{ row.datasetData?Object.values(row.datasetData[row.roleValue]).join(', ') : '' }}</span>
                  </el-tooltip>
                </p>
              </div>
            </el-col>

          </el-row>

        </div>
      </el-col>

      <el-col :span="8">
        <div class="col job flex-center justify-center shadow">
          <h3 class="list-title">JOB</h3>

          <div v-if="jobStatus==='failed' || jobStatus==='success'" class="job-end-container flex flex-col flex-center">
            <i
              v-if="jobStatus === 'failed'"
              class="el-icon-circle-close job-icon icon-error"
              style="color: #ff6464;"/>
            <i
              v-else-if="jobStatus === 'success'"
              class="el-icon-circle-check job-icon icon-error"
              style="color: #24b68b;"/>
            <ul class="job-info flex space-around flex-wrap w-100">
              <li>
                <p class="name">status</p>
                <p class="value">{{ jobStatus }}</p>
              </li>
              <li v-if="elapsed">
                <p class="name">duration</p>
                <p class="value">{{ elapsed }}</p>
              </li>
              <li v-if="AUC">
                <p class="name overflow-ellipsis">best score(AUC)</p>
                <p class="value">{{ AUC }}</p>
              </li>
              <li v-if="ratio">
                <p class="name">ratio</p>
                <p class="value">{{ ratio }}</p>
              </li>
              <li v-if="count">
                <p class="name">count</p>
                <p class="value">{{ count }}</p>
              </li>
            </ul>
          </div>

          <div v-else-if="jobStatus==='waiting' || jobStatus==='running'" class="echarts-container">
            <div v-if="elapsed" class="elapsed">
              <p class="elapsed-title">elapsed</p>
              <p class="elapsed-time text-primary">{{ elapsed }}</p>
            </div>
            <echart-container :class="'echarts'" :options="jobOptions" @getEchartInstance="getJobEchartInstance"/>
          </div>

          <div class="btn-wrapper flex justify-center">
            <el-button
              type="primary"
              round
              @click="toDetails(jobId)"
            >VIEW THIS JOB
            </el-button>
            <el-button
              v-show="jobStatus==='running'"
              type="primary"
              round
              @click="killJob"
            >KILL
            </el-button>
          </div>
        </div>
      </el-col>

      <el-col :span="8">
        <div v-loading="false" class="col graph flex-center justify-center shadow">
          <h3 class="list-title">GRAPH</h3>
          <div
            v-if="DAGData"
            :style="{'min-height':DAGData.component_list.length * 60+'px'}"
            class="wrapper w-100"
            style="min-width:400px;">
            <echart-container
              :class="'echarts'"
              :options="graphOptions"
              @getEchartInstance="getGraphEchartInstance"
            />
          </div>
        </div>
      </el-col>
    </el-row>

    <div class="log-wrapper shadow">
      <h3 class="title">LOG</h3>
      <ul class="tab-bar flex">
        <li
          v-for="(tab,index) in Object.keys(logsMap)"
          :key="index"
          :class="{'tab-btn-active':currentLogTab === tab}"
          class="tab-btn"
          @click="switchLogTab(tab)"
        >
          <span class="text">{{ tab }}</span>
          <span v-if="tab!=='all'" :class="[tab]" class="count">{{ logsMap[tab].length }}</span>
        </li>
        <!--<div class="tab-search">debug</div>-->
      </ul>
      <div v-loading="logLoading" ref="logView" class="log-container" @mousewheel="logOnMousewheel">
        <ul class="log-list overflow-hidden">
          <li v-for="(log,index) in logsMap[currentLogTab].list" :key="index">
            <div class="flex">
              <span style="color:#999;margin-right: 5px;">{{ log.lineNum }}</span>
              <span> {{ log.content }}</span>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import EchartContainer from '@/components/EchartContainer'
import jobOptions from '@/utils/chart-options/gauge'
import graphOptions from '@/utils/chart-options/graph'
import graphChartHandler from '@/utils/vendor/graphChartHandler'
import { formatSeconds, initWebSocket } from '@/utils'

import { getJobDetails, getDAGDpencencies, queryLog, killJob } from '@/api/job'

export default {
  components: {
    EchartContainer
  },
  data() {
    return {
      jobOptions,
      graphOptions,
      datasetList: [],
      jobId: this.$route.query.job_id,
      role: this.$route.query.role,
      partyId: this.$route.query.party_id,
      jobStatus: '',
      datasetLoading: true,
      logLoading: false,
      jobTimer: null,
      logWebsocket: {
        // 'all': null,
        'error': null,
        'warning': null,
        'info': null,
        'debug': null
      },
      jobWebsocket: null,
      logsMap: {
        // 'all': { list: [], length: 0 },
        'error': { list: [], length: 0 },
        'warning': { list: [], length: 0 },
        'info': { list: [], length: 0 },
        'debug': { list: [], length: 0 }
      },
      DAGData: null,
      gaugeInstance: null,
      graphInstance: null,
      ratio: '',
      count: '',
      AUC: '',
      elapsed: '',
      currentLogTab: 'info'
    }
  },
  mounted() {
    // console.log(process.env.BASE_API)
    this.getDatasetInfo()
    this.getDAGDpendencies()
    this.getLogSize()
    this.openLogsWebsocket()
    this.openJobWebsocket()
  },
  beforeDestroy() {
    clearInterval(this.jobTimer)
    this.closeWebsocket()
  },
  methods: {
    getDAGDpendencies() {
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId
      }
      getDAGDpencencies(para).then(res => {
        this.DAGData = res.data
      })
    },
    openLogsWebsocket() {
      Object.keys(this.logsMap).forEach(item => {
        this.logWebsocket[item] = initWebSocket(`/log/${this.jobId}/${this.role}/${this.partyId}/default/${item}`, res => {
          // console.log(item, 'success')
        }, res => {
          const data = JSON.parse(res.data)
          // console.log(item, data)
          if (Array.isArray(data)) {
            if (data.length > 0) {
              this.logsMap[item].list = [...this.logsMap[item].list, ...data]
              this.logsMap[item].length = data[data.length - 1].lineNum
            }
          } else {
            this.logsMap[item].list.push(data)
            this.logsMap[item].length = data.lineNum
          }
        })
      })
    },
    openJobWebsocket() {
      this.jobWebsocket = initWebSocket(`/websocket/progress/${this.jobId}/${this.role}/${this.partyId}`, res => {
        // console.log('job wbsocket success')
      }, res => {
        const { process, status, duration, dependency_data } = JSON.parse(res.data)
        // console.log(JSON.parse(res.data))
        if (this.graphInstance) {
          this.pushDataToGraphInstance(this.graphInstance, dependency_data.data)
        }
        if (duration) {
          this.elapsed = formatSeconds(duration)
        }
        this.jobStatus = status
        if (this.jobStatus !== 'failed' && this.jobStatus !== 'success') {
          this.jobOptions.series[0].pointer.show = true
          this.jobOptions.series[0].detail.show = true
          this.jobOptions.series[0].data[0].value = process || 0
        }
        if (this.gaugeInstance) {
          this.gaugeInstance.setOption(this.jobOptions, true)
        }
      })
    },
    getLogSize() {
      // Object.keys(this.logsMap).forEach(item => {
      //   queryLogSize({
      //     job_id: this.jobId,
      //     party_id: this.partyId,
      //     role: this.role,
      //     type: item
      //   }).then(res => {
      //     this.logsMap[item].length = res.data
      //   })
      // })
    },
    getDatasetInfo() {
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId
      }
      getJobDetails(para).then(res => {
        const { job, dataset: _dataset } = res.data
        if (_dataset) {
          const { roles, dataset } = _dataset

          Object.keys(roles).forEach(role => {
            const options = []
            roles[role].forEach(item => {
              options.push({
                value: item,
                label: item
              })
            })
            // if (dataset[role]) {
            this.datasetList.push({
              role: role.toUpperCase(),
              options,
              roleValue: options[0].label,
              datasetData: dataset[role] || ''
            })
            // }
          })
        }
        if (job) {
          this.jobStatus = job.fStatus
        }
      }).then(res => {
        this.datasetLoading = false
      })
    },
    getJobEchartInstance(echartInstance) {
      this.gaugeInstance = echartInstance
    },

    closeWebsocket() {
      // console.log('close Websocket')
      Object.keys(this.logWebsocket).forEach(type => {
        if (this.logWebsocket[type]) {
          this.logWebsocket[type].close()
        }
      })
      if (this.jobWebsocket) {
        this.jobWebsocket.close()
      }
    },

    killJob() {
      // console.log(this.jobWebsocket)
      this.$confirm('You can\'t undo this action', 'Are you sure you want to kill this job?', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }).then(() => {
        killJob({
          job_id: this.jobId,
          role: this.role,
          party_id: this.partyId
        })
        // console.log('kill job:' + this.jobId, this.role, this.partyId)
        // this.jobStatus = 'failed'
      }).catch(() => {
        // console.log('cancel kill')
      })
    },

    getGraphEchartInstance(echartInstance) {
      let fnInterval = null
      const fn = () => {
        if (this.DAGData) {
          window.clearInterval(fnInterval)
          // const { dataList, linksList } = graphChartHandler(this.DAGData)
          // console.log(dataList, linksList)
          // this.graphOptions.series[0].data = dataList
          // this.graphOptions.series[0].links = linksList
          this.graphOptions.tooltip.show = false
          // echartInstance.setOption(this.graphOptions, true)
          // echartInstance.on('click', { dataType: 'node' }, nodeData => {
          //   console.log(nodeData)
          // })
          this.pushDataToGraphInstance(echartInstance, this.DAGData)
        }
      }
      fnInterval = window.setInterval(fn, 100)
      this.graphInstance = echartInstance
    },
    pushDataToGraphInstance(instance, data) {
      const { dataList, linksList } = graphChartHandler(data)
      // console.log(dataList, linksList)
      this.graphOptions.series[0].data = dataList
      this.graphOptions.series[0].links = linksList
      instance.setOption(this.graphOptions, true)
    },
    toDetails() {
      this.$router.push({
        path: '/details',
        query: { job_id: this.jobId, role: this.role, party_id: this.partyId, 'from': 'Dashboard' }
      })
    },
    switchLogTab(tab) {
      this.currentLogTab = tab
    },
    logOnMousewheel(e) {
      // console.log(e.target.parentNode.parentNode.scrollTop)
      // console.log(e.wheelDelta)
      const topLog = this.logsMap[this.currentLogTab].list[0]
      if (!topLog) {
        return
      }
      const end = topLog.lineNum - 1
      if (end > 0) {
        const maxLoadingPage = 1000
        if (this.$refs['logView'].scrollTop === 0 && (e.wheelDelta > 0 || e.detail > 0)) {
          const begin = end - maxLoadingPage > 1 ? end - maxLoadingPage : 1
          if (!this.logLoading) {
            this.logLoading = true

            const fn = () => {
              queryLog({
                componentId: 'default',
                job_id: this.jobId,
                role: this.role,
                party_id: this.partyId,
                begin,
                end,
                type: this.currentLogTab
              }).then(res => {
                // console.log(res)
                const newLogs = []
                res.data.map(log => {
                  // console.log(log)
                  if (log) {
                    newLogs.push(log)
                  }
                })
                this.logsMap[this.currentLogTab].list = [...newLogs, ...this.logsMap[this.currentLogTab].list]
                this.logLoading = false
              }).catch(() => {
                this.logLoading = false
              })
            }

            window.setTimeout(fn, 1000)
          }
        }
      }
    }
  }
}
</script>

<style lang="scss">
  @import "../../styles/dashboard";
</style>
