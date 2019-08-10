<template>
  <div class="app-container details-container bg-dark">

    <h3 class="app-title">
      <span class="text-primary pointer" @click="toPrevPage">{{ jobFrom }}</span> > {{ jobId }}
    </h3>

    <section class="section-wrapper">
      <div class="flex space-between" style="padding-top: 12px;">
        <h3 class="section-title">JOB SUMMARY</h3>

        <el-button
          type="primary"
          round
          style="height: 32px;line-height: 0;font-size: 16px;transform: translateY(-12px);"
          @click="$router.push({path:'/dashboard',query:{job_id:jobId,role,party_id:partyId}})">dashboard
        </el-button>
      </div>

      <div v-loading="summaryLoading" class="section-view job-summary shadow">
        <el-row>
          <el-col :span="10" class="section-col">
            <div class="text-col">
              <p class="prop">Job ID</p>
              <p class="prop">Status</p>
            </div>
            <div class="text-col">
              <p class="value text-primary">{{ jobId }}</p>
              <p class="value text-primary">{{ jobInfo.status }}</p>
            </div>
          </el-col>
          <el-col :span="7" class="section-col">
            <div class="text-col">
              <p class="prop">Role</p>
              <p class="prop">Party_ID</p>
              <p v-for="(item,index) in roleList" :key="index" class="prop">{{ item.role }}</p>
            </div>

            <div class="text-col">
              <p class="value">{{ role }}</p>
              <p class="value">{{ partyId }}</p>
              <!--<p class="value">test target</p>-->
              <div v-for="(item,index) in roleList" :key="index" class="flex flex-center">
                <p class="value">{{ item.datasetList.length }}</p>
                <el-popover
                  placement="right-start"
                  title=""
                  width="250"
                  trigger="click">
                  <div>
                    <el-row>
                      <el-col :span="8">
                        <p
                          style="margin-bottom: 8px;
                                font-weight: bold;
                                color: #7f7d8e;
                                height: 18px;
                                line-height: 18px;"
                        >{{ item.role }}</p>
                        <p v-for="(dataset,index) in item.datasetList" :key="index">{{ dataset.name }}</p>
                      </el-col>
                      <el-col :span="12" :offset="4">
                        <p
                          style="margin-bottom: 8px;
                                font-weight: bold;
                                color: #7f7d8e;
                                height: 18px;
                                line-height: 18px;"
                        >DATASET</p>
                        <p v-for="(dataset,index) in item.datasetList" :key="index">{{ dataset.dataset }}</p>
                      </el-col>
                    </el-row>
                  </div>
                  <p slot="reference" class="text-primary tip">view</p>
                </el-popover>
              </div>
            </div>
            <!--<div>-->
            <!--<div class="flex">-->
            <!--<div class="text-col">-->
            <!--<p class="value">{{ guest.dataset }}</p>-->
            <!--<p class="value">{{ guest.columns }}</p>-->
            <!--</div>-->
            <!--<div class="text-col">-->
            <!--<p class="value">{{ guest.target }}</p>-->
            <!--<p class="value">{{ guest.rows }}</p>-->
            <!--</div>-->
            <!--</div>-->
            <!--<div class="flex">-->
            <!--<div class="text-col">-->
            <!--<p class="value">{{ guest.partner }}</p>-->
            <!--<p class="value">{{ guest.pnr_dataset }}</p>-->
            <!--</div>-->
            <!--</div>-->
            <!--</div>-->
          </el-col>
          <el-col :span="7" class="section-col">
            <div class="text-col">
              <p class="prop">Submission Time</p>
              <p class="prop">Start Time</p>
              <p class="prop">End time</p>
              <p class="prop">Duration</p>
            </div>

            <div class="text-col">
              <p class="value">{{ jobInfo.submmissionTime }}</p>
              <p class="value">{{ jobInfo.startTime }}</p>
              <p class="value">{{ jobInfo.endTime }}</p>
              <p class="value">{{ jobInfo.duration }}</p>
            </div>
          </el-col>
        </el-row>
      </div>
    </section>

    <section class="section-wrapper">
      <h3 class="section-title">OUTPUTS FROM JOB</h3>
      <div class="output-wrapper shadow flex">
        <!--DAG-->
        <div class="dag-wrapper overflow-auto">
          <h4 class="output-title">Main Graph</h4>
          <p class="output-desc">Click component to view details</p>
          <!--<div v-if="DAGData" :style="{'min-height':DAGData.component_list.length * 120+'px'}" class="echart-wrapper">-->
          <div v-if="DAGData" class="echart-wrapper">
            <echart-container
              :class="'echarts'"
              :options="graphOptions"
              @getEchartInstance="getGraphEchartInstance"
            />
          </div>
        </div>

        <div v-loading="paraLoading" class="para-wrapper">
          <h4 class="para-title">Parameter</h4>
          <div v-loading="msgLoading" class="msg bg-dark">
            <div v-show="paraData" class="flex">
              <pre> {{ paraData }} </pre>
            </div>
          </div>
          <el-button
            :disabled="!componentName"
            type="primary"
            round
            style="height: 32px;line-height: 0;font-size: 16px;"
            @click="visualization"
          >
            view the outputs
          </el-button>
        </div>
      </div>
    </section>
    <el-dialog
      :visible.sync="outputVisible"
      :title="outputTitle"
      :close-on-click-modal="false"
      :show-close="false"
      :fullscreen="fullscreen"
      width="80%"
      top="10vh"
      @close="initOutput"
    >
      <div class="dialog-icons flex flex-center">
        <icon-hover-and-active
          :class-name="'img-wrapper'"
          :default-url="icons.normal.fullscreen"
          :hover-url="icons.hover.fullscreen"
          :active-url="icons.active.fullscreen"
          @clickFn="fullscreen = !fullscreen"
        />
        <icon-hover-and-active
          :class-name="'img-wrapper'"
          :default-url="icons.normal.close"
          :hover-url="icons.hover.close"
          :active-url="icons.active.close"
          @clickFn="outputVisible = false"
        />
      </div>
      <div class="tab-bar flex">
        <div :class="{'tab-btn-active':currentTab === 'model'}" class="tab-btn" @click="switchLogTab('model')">
          <span class="text">model output</span>
        </div>
        <div :class="{'tab-btn-active':currentTab === 'data'}" class="tab-btn" @click="switchLogTab('data')">
          <span class="text">data output</span>
        </div>
        <div :class="{'tab-btn-active':currentTab === 'log'}" class="tab-btn" @click="switchLogTab('log')">
          <span class="text">log</span>
        </div>
      </div>
      <section
        v-loading="metricLoading && modelLoading"
        :style="{height:fullscreen?'calc(100vh - 120px)':'70vh'}"
        class="section-wrapper"
        style="padding: 0 48px 48px;margin-bottom: 0;overflow: auto">
        <!--<h3 class="section-title">Visualization</h3>-->
        <div class="section-view" style="padding: 0">

          <div v-show="currentTab === 'model'" class="tab">
            <model-output
              ref="dialog"
              :metric-output-list="metricOutputList"
              :evaluation-output-list="evaluationInstances"
              :model-summary-data="modelSummaryData"
              :model-output-type="modelOutputType"
              :model-output="modelOutputData"
              :is-no-metric-output="isNoMetricOutput"
              :is-no-model-output="isNoModelOutput"
            />
          </div>
          <div v-show="currentTab === 'data'">
            <data-output :t-header="dataOutputHeader" :t-body="dataOutputBody" :no-data="dataOutputNoData"/>
          </div>
          <div v-show="currentTab === 'log'" ref="logView">
            <ul
              :style="{height:fullscreen?'75vh':'63vh'}"
              class="log-list"
              @mousewheel="logOnMousewheel"
            >
              <li v-for="(log,index) in logList" :key="index" class="flex">
                <span class="num">{{ log.lineNum }}</span>
                <span>{{ log.content }}</span>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </el-dialog>
  </div>
</template>

<script>
import { parseTime, formatSeconds, initWebSocket } from '@/utils'
import { getJobDetails, getDAGDpencencies, getComponentPara, queryLog } from '@/api/job'
import { getMetrics, getMetricData, getDataOutput, getModelOutput } from '@/api/chart'
import EchartContainer from '@/components/EchartContainer'
import IconHoverAndActive from '@/components/IconHoverAndActive'
import graphChartHandler from '@/utils/vendor/graphChartHandler'
import modelOutputDataHandler from '@/utils/vendor/modelOutputHandler'
import metricDataHandle from '@/utils/vendor/metricOutputHandler'
import ModelOutput from './ModelOutput'
import DataOutput from './DataOutput'
import graphOptions from '@/utils/chart-options/graph'
// import stackBarOptions from '@/utils/chart-options/stackBar'
import treeOptions from '@/utils/chart-options/tree'
import lineOptions from '@/utils/chart-options/line'
// import KSOptions from '@/utils/chart-options/KS'
import doubleBarOptions from '@/utils/chart-options/doubleBar'

// import axios from 'axios'
import { mapGetters } from 'vuex'

export default {
  name: 'JobDtails',
  components: {
    EchartContainer,
    ModelOutput,
    DataOutput,
    IconHoverAndActive
  },
  data() {
    return {
      // job summary模块
      jobId: this.$route.query.job_id,
      role: this.$route.query.role,
      partyId: this.$route.query.party_id,
      jobFrom: this.$route.query.from,
      summaryLoading: true,
      msgLoading: false,
      paraData: null,
      fullscreen: false,
      roleList: [],
      jobInfo: {},
      componentName: '',
      logLoading: false,
      dagInstance: null,
      graphOptions,
      treeOptions,
      lineOptions,
      doubleBarOptions,
      outputGraphOptions: graphOptions,
      paraLoading: false,
      DAGData: null,
      modelSummaryData: {
        tHeader: [],
        tBody: []
      },
      outputVisible: false,
      metricOutputList: [],
      modelOutputType: '',
      modelOutputData: null,
      dataOutputHeader: [],
      dataOutputBody: [],
      dataOutputNoData: false,
      isNoMetricOutput: false,
      metricLoading: false,
      modelLoading: false,
      isNoModelOutput: false,
      outputTitle: '',
      currentTab: 'model',
      logList: [],
      logWebsocket: null
    }
  },
  computed: {
    ...mapGetters([
      'modelNameMap',
      'metricTypeMap',
      'icons',
      'evaluationInstances'
    ])
  },
  mounted() {
    this.getDatasetInfo()
    const para = {
      job_id: this.jobId,
      role: this.role,
      party_id: this.partyId
    }
    getDAGDpencencies(para).then(res => {
      this.DAGData = res.data
    })
  },
  beforeDestroy() {
    this.closeWebsocket()
  },
  methods: {
    getDatasetInfo() {
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId
      }
      getJobDetails(para).then(res => {
        this.summaryLoading = false
        const { job, dataset: _dataset } = res.data
        if (_dataset) {
          const { roles, dataset } = _dataset
          Object.keys(roles).forEach(role => {
            const datasetList = []
            roles[role].forEach(name => {
              let set = ''
              if (dataset[role]) {
                set = Object.values(dataset[role][name]).join(', ')
              }
              datasetList.push({
                name,
                dataset: set
              })
            })
            this.roleList.push({
              role: role.toUpperCase(),
              datasetList
            })
          })
        }
        if (job) {
          this.jobInfo = {
            submmissionTime: job.fCreateTime ? parseTime(new Date(job.fCreateTime)) : '',
            startTime: job.fStartTime ? parseTime(new Date(job.fStartTime)) : '',
            endTime: job.fEndTime ? parseTime(new Date(job.fEndTime)) : '',
            duration: job.fElapsed ? formatSeconds(job.fElapsed) : '',
            status: job.fStatus ? job.fStatus : ''
          }
        }
      })
    },
    toPrevPage() {
      console.log(this.$route)
      let path = null
      if (this.jobFrom === 'Job overview') {
        path = '/history'
      } else if (this.jobFrom === 'Dashboard') {
        path = '/dashboard'
      }
      this.$router.push({
        path,
        query: { job_id: this.jobId, role: this.role, party_id: this.partyId }
      })
    },

    getGraphEchartInstance(echartInstance) {
      this.dagInstance = echartInstance
      let fnInterval = null
      const fn = () => {
        if (this.DAGData) {
          window.clearInterval(fnInterval)
          const { dataList, linksList } = graphChartHandler(this.DAGData)
          // console.log(this.DAGData)
          this.graphOptions.series[0].data = dataList
          this.graphOptions.series[0].links = linksList
          echartInstance.setOption(this.graphOptions, true)
          // 点击交互
          echartInstance.on('click', { dataType: 'node' }, nodeData => {
            // console.log(nodeData)
            this.clickComponent(nodeData.name, nodeData.dataIndex, nodeData.data.componentType)
          })
        }
      }
      fnInterval = window.setInterval(fn, 100)
    },

    clickComponent(component_name, dataIndex, componentType) {
      this.componentName = component_name
      this.modelOutputType = componentType || ''
      this.outputTitle = this.modelOutputType ? `${componentType}: ${component_name}` : ''
      this.clickComponentChangeStyle(this.graphOptions.series[0].data, dataIndex)
      this.dagInstance.setOption(this.graphOptions)
      this.paraLoading = true
      this.getParams(component_name)
    },
    clickComponentChangeStyle(obj, dataIndex) {
      obj.forEach(item => {
        // item.itemStyle = {}
        item.label = item.sourceLabel
      })
      // obj[dataIndex].itemStyle = { color: '#494ece' }
      obj[dataIndex].label = { color: '#fff', backgroundColor: '#494ece' }
    },
    visualization() {
      this.initOutput()
      this.getMetrics(this.componentName)
      this.getModelOutput(this.componentName)
      // this.getDataOutput(this.componentName)
      this.outputVisible = true
    },
    initOutput() {
      this.metricOutputList = []
      // this.evaluationOutputList = []
      this.$store.dispatch('SetCurveInstances', [])
      if (this.$refs.dialog) {
        this.$refs.dialog.clearEchartInstance()
      }
      this.isNoMetricOutput = false
      this.isNoModelOutput = false
      this.metricLoading = false
      this.modelLoading = false
      this.modelOutputData = null
      this.fullscreen = false
      this.modelSummaryData = {
        tHeader: [],
        tBody: []
      }
      this.dataOutputHeader = []
      this.dataOutputBody = []
      this.dataOutputNoData = false
      this.currentTab = 'model'
      this.logList = []
      this.$store.dispatch('InitModelOutput')
      this.closeWebsocket()
    },
    switchLogTab(tab) {
      this.currentTab = tab
      if (tab === 'data' && this.dataOutputHeader.length === 0) {
        this.getDataOutput(this.componentName)
      }
      if (tab === 'log' && !this.logWebsocket) {
        this.logWebsocket = initWebSocket(`/log/${this.jobId}/${this.role}/${this.partyId}/${this.componentName}/default`, res => {
          // console.log('log websocket success')
        }, res => {
          // console.log('websocket data:', JSON.parse(res.data))
          const data = JSON.parse(res.data)
          if (Array.isArray(data)) {
            if (data.length > 0) {
              this.logList = [...this.logList, ...data]
            }
          } else {
            this.logList.push(data)
          }
        })
      }
    },
    logOnMousewheel(e) {
      // console.log(e.target.parentNode.parentNode.scrollTop)
      // console.log(e.wheelDelta)
      const topLog = this.logList[0]
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
                componentId: this.componentName,
                job_id: this.jobId,
                role: this.role,
                party_id: this.partyId,
                begin,
                end
              }).then(res => {
                const newLogs = []
                res.data.map(log => {
                  newLogs.push(log)
                })
                this.logList = [...newLogs, ...this.logList]
                this.logLoading = false
              }).catch(() => {
                this.logLoading = false
              })
            }

            window.setTimeout(fn, 1000)
          }
        }
      }
    },
    closeWebsocket() {
      // console.log('close Websocket')
      if (this.logWebsocket) {
        this.logWebsocket.close()
        this.logWebsocket = null
      }
    },
    getParams(component_name) {
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId,
        component_name
      }
      getComponentPara(para).then(res => {
        this.paraLoading = false
        this.paraData = JSON.stringify(res.data, null, 2)
      }).catch(() => {
        this.paraLoading = false
        this.paraData = 'NO DATA'
      })
    },
    getMetrics(component_name) {
      this.metricLoading = true
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId,
        component_name
      }
      getMetrics(para).then(res => {
        this.metricLoading = false
        const data = res.data
        if (data && Object.keys(data).length > 0) {
          const evaluationOutputList = []
          Object.keys(data).forEach((metric_namespace, nameSpaceIndex) => {
            data[metric_namespace].forEach((metric_name, nameIndex) => {
              const metricDataPara = {
                job_id: this.jobId,
                role: this.role,
                party_id: this.partyId,
                component_name,
                metric_namespace,
                metric_name
              }
              // const metricDataPara = Object.assign(para, { metric_namespace, metric_name })
              getMetricData(metricDataPara).then(res => {
                const { data, meta } = res.data
                if (data && meta) {
                  // data.map(item => {
                  //   item[1] = item[1].toFixed(4)
                  // })
                  const { metric_type, unit_name, curve_name, pair_type, thresholds } = meta
                  if (metric_type) {
                    metricDataHandle({
                      metricOutputList: this.metricOutputList,
                      modelSummaryData: this.modelSummaryData,
                      modelOutputType: this.modelOutputType,
                      evaluationOutputList,
                      data,
                      meta,
                      metric_type,
                      metric_namespace,
                      metric_name,
                      unit_name,
                      curve_name,
                      pair_type,
                      thresholds
                    })
                  }
                }
              })
              if (nameSpaceIndex === Object.keys(data).length - 1 && nameIndex === data[metric_namespace].length - 1) {
                setTimeout(() => {
                  const evaluationFlags = []
                  const arr = []
                  evaluationOutputList.forEach(item => {
                    if (item.type && arr.indexOf(item.type) === -1) {
                      arr.push(item.type)
                    }
                  })
                  // this.evaluationOutputList = evaluationOutputList
                  this.$store.dispatch('SetCurveInstances', evaluationOutputList)
                  const filterArr = ['ROC', 'K-S', 'Lift', 'Gain', 'Precision Recall', 'Accuracy'].filter(type => {
                    return arr.indexOf(type) !== -1
                  })
                  filterArr.forEach((item, index) => {
                    evaluationFlags.push(index === 0)
                  })
                  evaluationFlags[0] = true
                  this.$store.dispatch('SetCvFlags', evaluationFlags)
                  // evaluationOutputList.forEach(item => {
                  //   if (item.type === 'Precision Recall') {
                  //     console.log(item)
                  //   }
                  // })
                }, 1200)
              }
            })
          })
        } else {
          this.metricLoading = false
          this.isNoMetricOutput = true
        }
      })
    },
    getDataOutput(component_name) {
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId,
        component_name
      }
      getDataOutput(para).then(res => {
        const header = []
        const body = []
        if (res.data && res.data.meta && res.data.meta.header) {
          res.data.meta.header.forEach(item => {
            header.push({
              prop: item.replace('.', ''),
              label: item
            })
          })
          res.data.data.forEach(oldRow => {
            const newRow = {}
            header.forEach((item, index) => {
              let value = oldRow[index]
              if (typeof value === 'object') {
                value = JSON.stringify(value)
              }
              newRow[item.prop] = value && value.toString()
            })
            body.push(newRow)
          })
          if (header.length === 0 || body.length === 0) {
            this.dataOutputNoData = true
          } else {
            this.dataOutputHeader = header
            this.dataOutputBody = body
          }
        } else {
          this.dataOutputNoData = true
        }
      })
    },
    getModelOutput(component_name) {
      this.modelLoading = true
      const para = {
        job_id: this.jobId,
        role: this.role,
        party_id: this.partyId,
        component_name
      }
      getModelOutput(para).then(res => {
        this.modelLoading = false
        // this.modelOutputType = res.data.meta ? res.data.meta.module_name : ''
        // this.outputTitle = this.modelOutputType || ''
        // if (this.outputTitle) {
        //   this.outputTitle += ': '
        // }
        // this.outputTitle += component_name
        const responseData = res.data.data ? res.data.data : null
        this.modelOutputData = modelOutputDataHandler({
          outputType: this.modelOutputType,
          responseData,
          isNoModelOutput: this.isNoModelOutput
        })
        this.isNoModelOutput = Boolean(this.modelOutputData.isNoModelOutput)
      }).catch(() => {
        this.modelLoading = false
        this.isNoModelOutput = true
      })
    }
  }
}
</script>

<style lang="scss">
  @import "../../styles/details";
</style>
