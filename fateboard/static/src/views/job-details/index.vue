<template>
  <div class="app-container details-container bg-dark">

    <h3 class="app-title">
      <span class="text-primary pointer" @click="toPrevPage">{{ jobFrom }}</span> > {{ jobId }}
    </h3>

    <section class="section-wrapper">
      <div class="flex space-between" style="padding-top: 12px;">
        <h3 class="section-title">JOB SUMMARY</h3>
        <!--回到dashboard按钮-->
        <el-button
          type="primary"
          round
          style="transform: translateY(-12px);"
          @click="$router.push({path:'/dashboard',query:{jobId}})">dashboard
        </el-button>
      </div>

      <div v-loading="summaryLoading" class="section-view job-summary shadow">
        <el-row>
          <el-col :span="6" class="section-col">
            <div class="text-col">
              <p class="prop">Job ID</p>
              <p class="prop">Status</p>
            </div>
            <div class="text-col">
              <p class="value" style="color: #494ece">{{ jobId }}</p>
              <p class="value" style="color: #494ece">{{ status }}</p>
            </div>
          </el-col>
          <el-col :span="9" class="section-col">
            <div class="text-col">
              <p class="prop">Guest</p>
              <p class="prop" style="margin-top: 34px">Host</p>
            </div>
            <div>
              <div class="flex">
                <div class="text-col">
                  <p class="value">{{ guest.dataset }}</p>
                  <p class="value">{{ guest.columns }}</p>
                </div>
                <div class="text-col">
                  <p class="value">{{ guest.target }}</p>
                  <p class="value">{{ guest.rows }}</p>
                </div>
              </div>
              <div class="flex">
                <div class="text-col">
                  <p class="value">{{ guest.partner }}</p>
                  <p class="value">{{ guest.pnr_dataset }}</p>
                </div>
              </div>
            </div>
          </el-col>
          <el-col :span="9" class="section-col">
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
        <div class="dag-wrapper">
          <h4 class="output-title">Main Ggraph</h4>
          <p class="output-desc">Click component to view details</p>
          <div v-if="DAGData" :style="{'min-height':DAGData.componentList.length * 60+'px'}" class="echart-wrapper">
            <echart-container
              :class="'echarts'"
              :options="graphOptions"
              @getEchartInstance="getGraphEchartInstance"
            />
          </div>
        </div>

        <div v-loading="paraLoading" class="para-wrapper">
          <h4 class="para-title">Parameter({{ paraData ?Object.keys(paraData).length:0 }})</h4>
          <div v-loading="msgLoading" class="msg bg-dark">
            <div v-show="paraData" class="flex">
              <pre> {{ JSON.stringify(paraData, null, 2) }} </pre>
            </div>
          </div>
          <el-button
            :disabled="!componentName"
            type="primary"
            round
            @click="visualization"
          >
            visualization
          </el-button>
        </div>
      </div>
    </section>

    <section v-if="modelSummaryData" class="section-wrapper">
      <h3 class="section-title">MODEL SUMMARY</h3>
      <div class="section-view flex shadow">
        <el-table :data="modelSummaryData.data" style="width: 100%;">
          <el-table-column
            v-for="(item,index) in modelSummaryData.header"
            :key="index"
            :prop="item.prop"
            :label="item.label"
            show-overflow-tooltip
            align="center"
          />
        </el-table>
      </div>
    </section>

    <el-dialog
      :visible.sync="outputVisible"
      :title="outputTitle"
      width="80%"
      top="10vh"
      @open="modelOutputLoading = true"
      @close="initOutput"
    >

      <section v-loading="modelOutputLoading" class="section-wrapper" style="padding: 0">
        <!--<h3 class="section-title">Visualization</h3>-->
        <div class="section-view" style="padding: 0">

          <div class="tab-bar flex">
            <div :class="{'tab-btn-active':currentTab === 'model'}" class="tab-btn" @click="switchLogTab('model')">
              <span class="text">model_output</span>
            </div>
            <div :class="{'tab-btn-active':currentTab === 'data'}" class="tab-btn" @click="switchLogTab('data')">
              <span class="text">data_output</span>
            </div>
            <div :class="{'tab-btn-active':currentTab === 'log'}" class="tab-btn" @click="switchLogTab('log')">
              <span class="text">log</span>
            </div>
          </div>

          <div v-show="currentTab === 'model'" class="tab">
            <model-output
              :metric-output-list="metricOutputList"
              :model-output-type="modelOutputType"
              :model-output="modelOutput"
            />
          </div>
          <div v-show="currentTab === 'data'">
            <data-output :t-header="dataOutputHeader" :t-body="dataOutputBody"/>
          </div>
          <div v-show="currentTab === 'log'">
            <ul v-loading="logLoading" class="log-list" @mousewheel="logOnMousewheel">
              <li v-for="(log,index) in logList" :key="index">
                <span style="color:#999;margin-right: 5px;">{{ log.lineNum }}</span>
                {{ log.content }}
              </li>
            </ul>
          </div>
        </div>
      </section>
    </el-dialog>
  </div>
</template>

<script>
import { parseTime, formatSeconds, jsonToTableHeader, deepClone, initWebSocket } from '@/utils'
import { getJobDetails, getDAGDpencencies, getComponentPara, queryLog } from '@/api/job'
import { getMetrics, getMetricData, getDataOutput, getModelOutput } from '@/api/chart'
import EchartContainer from '@/components/EchartContainer'
import graphChartHandle from '@/utils/vendor/graphChartHandle'
import ModelOutput from './ModelOutput'
import DataOutput from './DataOutput'
import graphOptions from '@/utils/chart-options/graph'
import stackBarOptions from '@/utils/chart-options/stackBar'
import treeOptions from '@/utils/chart-options/tree'
import lineOptions from '@/utils/chart-options/line'
import KSOptions from '@/utils/chart-options/KS'
import doubleBarOptions from '@/utils/chart-options/doubleBar'

// import axios from 'axios'

export default {
  name: 'JobDtails',
  components: {
    EchartContainer,
    ModelOutput,
    DataOutput
  },
  data() {
    return {
      // job summary模块
      jobId: this.$route.query.jobId,
      jobFrom: this.$route.query.from,
      status: 'complete',
      summaryLoading: true,
      msgLoading: false,
      paraData: null,
      guest: {},
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
      modelSummaryData: null,
      outputVisible: false,
      metricOutputList: [],
      modelOutputLoading: false,
      modelOutputType: '',
      modelOutput: null,
      dataOutputHeader: [],
      dataOutputBody: [],
      outputTitle: '',
      currentTab: 'model',
      logList: [],
      logWebsocket: null
    }
  },
  mounted() {
    this.getDatasetInfo()
    getDAGDpencencies(this.jobId).then(res => {
      this.DAGData = res.data
    })

    // axios.post('http://172.16.153.113:16688/v1/tracking/component/output/data',
    //   {
    //     job_id: this.jobId,
    //     component_name: 'P0001E0001T007'
    //   }
    // )
  },
  beforeDestroy() {
    this.closeWebsocket()
  },
  methods: {
    getDatasetInfo() {
      getJobDetails(this.jobId).then(res => {
        this.summaryLoading = false
        const { job, dataset } = res.data
        this.guest = {
          dataset: dataset.dataset,
          target: dataset.target,
          rows: dataset.row,
          columns: dataset.columns,
          partner: dataset.partner,
          pnr_dataset: dataset.pnr_dataset
        }
        this.jobInfo = {
          submmissionTime: parseTime(new Date(job.fCreateTime)),
          startTime: parseTime(new Date(job.fStartTime)),
          endTime: parseTime(new Date(job.fEndTime)),
          duration: formatSeconds(3800)
        }

        this.modelSummaryData = jsonToTableHeader(dataset.model_summary, '')
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
        query: { jobId: this.jobId }
      })
    },

    getGraphEchartInstance(echartInstance) {
      this.dagInstance = echartInstance
      let fnInterval = null
      const fn = () => {
        if (this.DAGData) {
          window.clearInterval(fnInterval)
          const { dataList, linksList } = graphChartHandle(this.DAGData)
          // console.log(this.DAGData)
          this.graphOptions.series[0].data = dataList
          this.graphOptions.series[0].links = linksList
          echartInstance.setOption(this.graphOptions, true)
          // 点击交互
          echartInstance.on('click', { dataType: 'node' }, nodeData => {
            // console.log(nodeData)
            this.clickComponent(nodeData.name, nodeData.dataIndex)
          })
        }
      }
      fnInterval = window.setInterval(fn, 100)
    },

    clickComponent(component_name, dataIndex) {
      this.componentName = component_name
      this.clickComponentChangeStyle(this.graphOptions.series[0].data, dataIndex)
      this.dagInstance.setOption(this.graphOptions)
      // this.initOutput()
      this.paraLoading = true
      this.getParams(component_name)
      // this.getMetrics(component_name)
      // this.getModelOutput(component_name)
      // this.getDataOutput(component_name)
      // this.outputVisible = true
    },
    clickComponentChangeStyle(obj, dataIndex) {
      obj.forEach(item => {
        // item.itemStyle = {}
        item.label = {}
      })
      // obj[dataIndex].itemStyle = { color: '#494ece' }
      obj[dataIndex].label = { color: '#fff', backgroundColor: '#494ece' }
    },
    visualization() {
      this.initOutput()
      this.getMetrics(this.componentName)
      this.getModelOutput(this.componentName)
      this.getDataOutput(this.componentName)
      this.outputVisible = true
    },
    initOutput() {
      this.metricOutputList = []
      this.modelOutput = null
      this.modelOutputType = ''
      this.dataOutputHeader = []
      this.dataOutputBody = []
      this.currentTab = 'model'
      this.logList = []
      this.outputTitle = ''
      this.closeWebsocket()
    },
    switchLogTab(tab) {
      this.currentTab = tab
      if (tab === 'data' && this.dataOutputHeader.length === 0) {
        this.getDataOutput(this.componentName)
      }
      if (tab === 'log' && !this.logWebsocket) {
        this.logWebsocket = initWebSocket(`/log/${this.jobId}/${this.componentName}/default`, res => {
          // console.log('日志推送websocket连接成功')
        }, res => {
          // console.log('websocket请求回来的数据:', JSON.parse(res.data))
          this.logList.push(JSON.parse(res.data))
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
        if (e.target.parentNode.parentNode.scrollTop === 0 && (e.wheelDelta > 0 || e.detail > 0)) {
          // console.log('鼠标滚轮往上滑，加载前面的日志')
          const begin = end - 10 > 1 ? end - 10 : 1
          if (!this.logLoading) {
            this.logLoading = true

            const fn = () => {
              queryLog({
                componentId: this.componentName,
                jobId: this.jobId,
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
      console.log('close Websocket')
      if (this.logWebsocket) {
        this.logWebsocket.close()
        this.logWebsocket = null
      }
    },
    getParams(component_name) {
      getComponentPara({
        job_id: this.jobId,
        component_name: component_name
      }).then(res => {
        this.paraLoading = false
        this.paraData = res.data
      })
    },
    getMetrics(component_name) {
      getMetrics({
        job_id: this.jobId,
        component_name: component_name
      }).then(res => {
        const data = res.data
        if (data) {
          Object.keys(data).forEach(metric_namespace => {
            data[metric_namespace].forEach(metric_name => {
              getMetricData({
                job_id: this.jobId,
                component_name: component_name,
                metric_namespace,
                metric_name
              }).then(res => {
                this.modelOutputLoading = false
                const { data, meta } = res.data
                if (data && meta) {
                  const { metric_type, unit_name, curve_name } = meta
                  let type = ''
                  if (metric_type === 'LOSS') {
                    type = 'line'
                    const outputData = deepClone(lineOptions)
                    outputData.xAxis.data = Object.keys(data)
                    outputData.xAxis.name = unit_name
                    outputData.series[0].data = Object.values(data)
                    this.metricOutputList.push({
                      type,
                      nameSpace: metric_namespace,
                      data: outputData
                    })
                  }
                  // KS曲线
                  if (metric_type === 'KS') {
                    const outputData = deepClone(KSOptions)
                    outputData.xAxis.data = Object.keys(data)
                    outputData.xAxis.name = unit_name
                    outputData.series.push({
                      name: curve_name,
                      type: 'line',
                      symbol: 'none',
                      data: Object.values(data)
                    })
                    if (this.metricOutputList.length === 0) {
                      this.metricOutputList.push({
                        type: 'KS',
                        nameSpace: metric_namespace,
                        data: outputData
                      })
                    } else {
                      this.metricOutputList.forEach((item, index) => {
                        if (item.type === 'KS' && item.nameSpace === metric_namespace) {
                          this.metricOutputList[index].data.series.push({
                            name: curve_name,
                            type: 'line',
                            symbol: 'none',
                            data: Object.values(data)
                          })
                        } else {
                          this.metricOutputList.push({
                            type: 'KS',
                            nameSpace: metric_namespace,
                            data: outputData
                          })
                        }
                      })
                    }
                    console.log(this.metricOutputList)
                  }
                }
              })
            })
          })
        }
      })
    },
    getDataOutput(component_name) {
      getDataOutput({
        job_id: this.jobId,
        component_name: component_name
      }).then(res => {
        const header = []
        const body = []
        res.data.meta.header.forEach(item => {
          header.push({
            prop: item,
            label: item
          })
        })
        res.data.data.forEach(oldRow => {
          const newRow = {}
          res.data.meta.header.forEach((item, index) => {
            newRow[item] = oldRow[index]
          })
          body.push(newRow)
        })
        this.dataOutputHeader = header
        this.dataOutputBody = body
      })
    },
    getModelOutput(component_name) {
      getModelOutput({
        job_id: this.jobId,
        component_name: component_name
      }).then(res => {
        this.modelOutputLoading = false
        this.modelOutputType = res.data.meta ? res.data.meta.module_name : ''
        this.outputTitle = this.modelOutputType || ''
        if (this.outputTitle) {
          this.outputTitle += ': '
        }
        this.outputTitle += component_name
        const responseData = res.data.data ? res.data.data : ''
        if (this.modelOutputType === 'HeteroSecureBoost') {
          this.modelOutput = {
            formatObj: responseData,
            formatString: JSON.stringify(responseData, null, 2)
          }
          // dataio
        } else if (this.modelOutputType === 'DataIO') {
          const imputerData = []
          const outlierData = []
          const { imputer_param, outlier_param } = responseData
          Object.keys(imputer_param.missingReplaceValue).forEach(key => {
            imputerData.push({
              variable: key,
              method: imputer_param.strategy,
              value: imputer_param.missingReplaceValue[key]
            })
          })
          Object.keys(outlier_param.outlierReplaceValue).forEach(key => {
            outlierData.push({
              variable: key,
              method: outlier_param.strategy,
              value: outlier_param.outlierReplaceValue[key]
            })
          })
          this.modelOutput = {
            imputerData,
            outlierData
          }
        } else if (this.modelOutputType === 'Intersection') {
          this.modelOutput = responseData
        } else if (this.modelOutputType === 'FeatureScale') {
          let tHeader = []
          const tData = []
          let data = null
          let method = responseData.method
          if (method === 'min_max_scale') {
            data = responseData.min_max_scale_param
            tHeader = [
              {
                prop: 'min',
                label: 'min'
              },
              {
                prop: 'max',
                label: 'max'
              },
              {
                prop: 'range_min',
                label: 'range_min'
              },
              {
                prop: 'range_max',
                label: 'range_max'
              }
            ]
          } else {
            data = responseData.standardScaleParam
            method = 'standard_scale'
            tHeader = [
              {
                prop: 'mean',
                label: 'mean'
              },
              {
                prop: 'scale',
                label: 'std'
              }
            ]
          }
          if (data) {
            Object.keys(data).forEach(key => {
              let item = {
                variable: key
              }
              tHeader.forEach((header, index) => {
                if (Array.isArray(data[key])) {
                  item[header.prop] = data[key][index]
                } else {
                  item = Object.assign(item, data[key])
                }
              })
              tData.push(item)
            })
            this.modelOutput = {
              method,
              tHeader,
              tData
            }
          }
        } else if (this.modelOutputType === 'HeteroLR' || this.modelOutputType === 'HomoLR') {
          const { weight, intercept, converged } = responseData
          const tData = []
          Object.keys(weight).forEach(key => {
            tData.push({
              variable: key,
              weight: weight[key]
            })
          })
          tData.push({
            variable: 'intercept',
            weight: intercept
          })
          this.modelOutput = {
            tData,
            converged
          }
        } else if (this.modelOutputType === 'HeteroFeatureSelection') {
          const data = responseData.results
          const chartData = []
          data.forEach(item => {
            const { filterName, featureValues } = item
            if (filterName && featureValues) {
              const options = deepClone(doubleBarOptions)
              options.title.text = filterName
              options.yAxis.data = Object.keys(featureValues)
              const value = Object.values(featureValues)
              options.series[0].label.formatter = function(params) {
                return value[params.dataIndex]
              }
              let max = 0
              value.forEach(item => {
                if (item > max) {
                  max = item
                }
              })
              Object.keys(featureValues).forEach(() => {
                options.series[0].data.push(max * 1.2)
              })
              options.series[1].data = value
              chartData.push(options)
            }
          })
          this.modelOutput = { chartData }
        } else if (this.modelOutputType === 'HeteroFeatureBinning') {
          const sourceData = []
          const options = []
          const variableData = {}
          const stackBarData = {}
          const woeData = {}
          const data = responseData.binningResult.binningResult
          Object.keys(data).forEach(key => {
            const tableData = []
            let min = 0
            // 随便找的一个数组元素，主要为了拿到下标遍历
            data[key].ivArray.forEach((item, index, self) => {
              const point = data[key].splitPoints[index] || data[key].splitPoints[index - 1]
              let binning = ''
              if (min === 0) {
                binning = `a < ${point}`
              } else if (index === self.length - 1) {
                binning = `a >= ${point}`
              } else {
                binning = `${min} <= a < ${point}`
              }
              min = point
              tableData.push({
                binning,
                event_count: data[key].eventCountArray[index],
                event_ratio: data[key].eventRateArray[index],
                non_event_count: data[key].nonEventCountArray[index],
                non_event_ratio: data[key].nonEventRateArray[index],
                woe: data[key].woeArray[index],
                iv: data[key].ivArray[index]
              })
            })
            variableData[key] = tableData
            const eventOptions = deepClone(stackBarOptions)
            const woeOptions = deepClone(stackBarOptions)
            eventOptions.series.push({
              name: 'event count',
              type: 'bar',
              data: data[key].eventCountArray,
              stack: 'event'
              // barWidth: '20%',
            })

            eventOptions.series.push({
              name: 'non-event count',
              type: 'bar',
              data: data[key].nonEventCountArray,
              stack: 'event'
              // barWidth: '20%',
            })
            for (let i = 1; i <= data[key].eventCountArray.length; i++) {
              eventOptions.xAxis.data.push(i)
            }
            stackBarData[key] = eventOptions

            woeOptions.series.push({
              name: 'woe',
              type: 'bar',
              data: data[key].woeArray
              // barWidth: '20%',
            })
            woeOptions.series.push({
              name: '',
              type: 'line',
              data: data[key].woeArray
              // barWidth: '20%',
            })
            woeData[key] = woeOptions
            sourceData.push({
              variable: key,
              iv: data[key].iv,
              monotonicity: data[key].isWoeMonotonic ? data[key].isWoeMonotonic.toString() : ''
            })
            options.push({
              value: key,
              label: key
            })
          })
          this.modelOutput = {
            sourceData,
            options,
            variableData,
            stackBarData,
            woeData
          }
        } else if (component_name.includes('sample')) {
          this.modelOutputType = 'sample'
        } else if (component_name.includes('hot')) {
          this.modelOutputType = 'hot'
        } else if (component_name.includes('evaluation')) {
          this.modelOutputType = 'evaluation'
        }
      })
    }
  }
}
</script>

<style lang="scss">
  @import "../../styles/details";
</style>
