<template>

  <section>
    <div v-if="isNoMetricOutput && isNoModelOutput" class="no-data">No data</div>
    <ul>
      <li v-for="(output,index) in metricOutputList" :key="index">
        <div v-if="output.type==='text'" class="flex">
          <p class="model-text" v-html="output.data"/>
        </div>
        <div v-if="output.type==='table'" class="flex">
          <el-table
            :data="output.data.tBody"
            highlight-current-row
            fit
            border
          >
            <el-table-column v-if="output.index" :label="output.index.label" type="index" width="100px" align="center"/>
            <el-table-column
              v-for="(item,index) in output.data.tHeader"
              :key="index"
              :label="item.label"
              :prop="item.prop"
              align="center"
            />
          </el-table>
        </div>
      </li>
    </ul>
    <ul v-if="lossList.length>0" class="cv-wrapper">
      <li
        v-for="(output,instanceIndex) in lossList"
        :key="instanceIndex"
      >
        <div class="w-100 overflow-hidden">
          <div class="cv-top flex flex-center space-between">
            <div class="flex flex-center">
              <h3 style="margin-right: 20px;">{{ output.type }}</h3>
              <p>{{ output.nameSpace }}</p>
            </div>
            <curve-legend
              :legend-data="output.legendData"
              :instance-index="instanceIndex"
              :instance-list="lossInstanceList"
              @clickLegend="clickLegend"/>
          </div>
          <echart-container
            :class="'echart'"
            :options="output.data"
            :legend-index="instanceIndex"
            @getEchartInstance="getLossInstance"/>
        </div>
      </li>
    </ul>
    <div v-if="modelOutputType && modelOutput && !isNoModelOutput">
      <!--boost-->
      <div v-if="modelOutputType===modelNameMap.boost">
        <pre class="boost-json"> {{ modelOutput.formatString }} </pre>
      </div>
      <!--dataio-->
      <div v-else-if="modelOutputType===modelNameMap.dataIO">
        <div v-if="modelOutput.imputerData">
          <div class="flex flex-center space-between" style="padding: 10px">
            <h2>Missing Fill Detail</h2>
          </div>
          <pagination-table
            :table-data="modelOutput.imputerData"
            :page-size="10"
            :header="dataIoImputerHeader"
          />
        </div>
        <div v-if="modelOutput.outlierData">
          <div class="flex flex-center space-between" style="padding: 10px">
            <h2>Outlier Replace Detail</h2>
          </div>
          <pagination-table
            :table-data="modelOutput.outlierData"
            :page-size="10"
            :header="dataIoOulierHeader"
          />
        </div>
      </div>
      <!--scale-->
      <div v-else-if="modelOutputType===modelNameMap.scale">
        <pagination-table
          :table-data="modelOutput.tBody"
          :page-size="10"
          :header="filterScaleHeader"
        />
        <!--<p>method:{{ modelOutput.method }}</p>-->
        <!--<el-table-->
        <!--:data="modelOutput.tData"-->
        <!--highlight-current-row-->
        <!--fit-->
        <!--border-->
        <!--empty-text="NO DATA"-->
        <!--&gt;-->
        <!--<el-table-column type="index" label="index" width="100px" align="center"/>-->
        <!--<el-table-column label="variable" prop="variable" align="center"/>-->
        <!--<el-table-column-->
        <!--v-for="(item,index) in modelOutput.tHeader"-->
        <!--:key="index"-->
        <!--:label="item.label"-->
        <!--:prop="item.prop"-->
        <!--align="center"-->
        <!--/>-->
        <!--</el-table>-->
      </div>
      <!--lr-->
      <div v-else-if="(modelOutputType===modelNameMap.homoLR || modelOutputType===modelNameMap.heteroLR)">
        <p class="model-text" style="margin-bottom: 0">max iterations: {{ modelOutput.iters }}</p>
        <p class="model-text">converged: {{ modelOutput.isConverged }}</p>
        <pagination-table
          :table-data="modelOutput.tData"
          :page-size="10"
          :header="lrHeader"
        />
      </div>
      <!--selection-->
      <div v-else-if="modelOutputType===modelNameMap.selection">
        <ul class="selection-list flex space-between flex-wrap">
          <li v-for="(item,index) in modelOutput.chartData" :key="index">
            <div :style="{height:item.containerHeight+'px'}" class="w-100">
              <echart-container :class="'wh-100'" :options="item"/>
            </div>
          </li>
        </ul>
      </div>
      <!--one hot-->
      <div v-else-if="modelOutputType===modelNameMap.oneHot">
        <el-select v-model="oneHotSelectValue" @change="changeOneHot">
          <el-option
            v-for="(item,index) in modelOutput.options"
            :key="index"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <div v-if="oneHotSelectValue" style="margin-top: 20px;">
          <pagination-table
            :table-data="modelOutput.variableData[oneHotSelectValue]"
            :page-size="10"
            :header="oneHotHeader"
          />
        </div>
      </div>
      <!--binning-->
      <div v-else-if="modelOutputType===modelNameMap.binning">
        <div v-if="role==='guest'" class="flex flex-end" style="margin-bottom: 24px;">
          <el-radio v-model="binningType" label="guest">guest</el-radio>
          <el-radio v-model="binningType" label="host">host</el-radio>
        </div>
        <pagination-table
          :table-data="modelOutput[currentbinningData].sourceData"
          :page-size="10"
          :header="binningSummaryHeader"
        />
        <div style="border:1px solid #eee;padding: 30px;margin-top: 25px">
          <el-select v-model="binningSelectValue" @change="changeBinning">
            <el-option
              v-for="(item,index) in modelOutput[currentbinningData].options"
              :key="index"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <div v-if="binningSelectValue" style="margin-top: 20px;">
            <pagination-table
              :table-data="modelOutput[currentbinningData].variableData[binningSelectValue]"
              :page-size="10"
              :header="binningHeader"
            />
          </div>
        </div>
        <echart-container :class="'echart'" @getEchartInstance="getStackBarInstance"/>
        <echart-container :class="'echart'" @getEchartInstance="getWoeInstance"/>
      </div>
    </div>
    <!--model summary-->
    <div v-if="modelSummaryData.tHeader.length>0 && modelSummaryData.tBody.length>0" style="margin-bottom: 20px;">
      <h3 style="margin-bottom: 20px;">{{ modelSummaryTitle }}</h3>
      <el-table
        :data="modelSummaryData.tBody"
        :span-method="summarySpanMethod"
        highlight-current-row
        fit
        border
        empty-text="NO DATA"
      >
        <el-table-column
          v-for="(item,index) in modelSummaryData.tHeader"
          :key="index"
          :label="item.label"
          :prop="item.prop"
          align="center"
          show-overflow-tooltip
        />
      </el-table>
    </div>
    <!--cv-->
    <ul v-if="evaluationOutputTypeList.length>1" class="cv-tab-list flex flex-center">
      <li
        v-for="(type,index) in evaluationOutputTypeList"
        :key="index"
        :class="{active:currentCvTab===index}"
        @click="changeCvTab(index)"
      >{{ type }}
      </li>
    </ul>
    <ul v-if="evaluationInstances.length>0" class="cv-wrapper">
      <li
        v-for="(output,instanceIndex) in evaluationInstances"
        v-show="output.type === evaluationOutputTypeList[currentCvTab]"
        :key="instanceIndex"
      >
        <div v-if="haveDataTypeList.indexOf(output.type)!==-1" class="w-100 overflow-hidden">
          <div class="cv-top flex flex-center space-between">
            <div class="flex flex-center">
              <h3 style="margin-right: 20px;">{{ output.type }}</h3>
              <p>{{ output.nameSpace }}</p>
            </div>
            <curve-legend
              :legend-data="output.legendData"
              :instance-index="instanceIndex"
              :instance-list="echartInstanceList"
              @clickLegend="clickLegend"/>
          </div>
          <echart-container
            :class="'echart'"
            :options="output.data"
            :legend-index="instanceIndex"
            @getEchartInstance="getEchartInstance"/>
        </div>
      </li>
    </ul>
  </section>
</template>

<script>
import EchartContainer from '@/components/EchartContainer'
import PaginationTable from './PaginationTable'
import { deepCloneArr } from '@/utils'
import CurveLegend from './CurveLegend'
import { mapGetters } from 'vuex'

export default {
  name: 'ModelOutput',
  components: {
    EchartContainer,
    PaginationTable,
    CurveLegend
  },
  props: {
    metricOutputList: {
      type: Array,
      default() {
        return []
      }
    },
    modelSummaryData: {
      type: Object,
      default() {
        return {
          tHeader: [],
          tBody: []
        }
      }
    },
    role: {
      type: String,
      default: ''
    },
    modelOutputType: {
      type: String,
      default: ''
    },
    modelOutput: {
      type: Object,
      default() {
        return {}
      }
    },
    isNoModelOutput: {
      type: Boolean,
      default: false
    },
    isNoMetricOutput: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      binningSelectValue: '',
      oneHotSelectValue: '',
      stackBarInstance: null,
      echartInstanceList: [],
      lossInstanceList: [],
      woeInstance: null,
      imputerDataPage: 1,
      outlierDataPage: 1,
      binningType: 'guest',
      lazyBinningType: 'guest',
      evaluationOutputTypeListArr: ['ROC', 'K-S', 'Lift', 'Gain', 'Precision Recall', 'Accuracy'],
      dataIoOulierHeader: [
        {
          prop: 'variable',
          label: 'variable'
        },
        {
          prop: 'ratio',
          label: 'outlier value ratio'
        },
        {
          prop: 'value',
          label: 'fill_value'
        }
      ],
      oneHotHeader: [
        {
          prop: 'value',
          label: 'value'
        },
        {
          prop: 'encoded_vector',
          label: 'encoded_vector'
        }
      ],
      lrHeader: [
        {
          prop: 'variable',
          label: 'variable'
        },
        {
          prop: 'weight',
          label: 'weight'
        }
      ],
      scaleHeader: [
        {
          prop: 'variable',
          label: 'variable'
        },
        {
          prop: 'columnLower',
          label: 'columnLower'
        },
        {
          prop: 'columnUpper',
          label: 'columnUpper'
        },
        {
          prop: 'mean',
          label: 'mean'
        },
        {
          prop: 'std',
          label: 'std'
        }
      ],
      dataIoImputerHeader: [
        {
          prop: 'variable',
          label: 'variable'
        },
        {
          prop: 'ratio',
          label: 'imputer value ratio'
        },
        {
          prop: 'value',
          label: 'fill_value'
        }
      ],
      binningSummaryHeader: [
        {
          prop: 'variable',
          label: 'variable'
        },
        {
          prop: 'iv',
          label: 'IV'
        },
        {
          prop: 'monotonicity',
          label: 'monotonicity'
        }
      ],
      binningHeader: [
        {
          prop: 'binning',
          label: 'binning'
        },
        {
          prop: 'iv',
          label: 'iv'
        },
        {
          prop: 'woe',
          label: 'woe'
        },
        {
          prop: 'event_count',
          label: 'event_count'
        },
        {
          prop: 'event_ratio',
          label: 'event_ratio'
        },
        {
          prop: 'non_event_count',
          label: 'non_event_count'
        },
        {
          prop: 'non_event_ratio',
          label: 'non_event_ratio'
        }
      ]
    }
  },

  computed: {
    ...mapGetters([
      'modelNameMap',
      'metricTypeMap',
      'currentCvTab',
      'evaluationFlags',
      'evaluationInstances'
    ]),
    evaluationOutputTypeList() {
      const arr = []
      this.evaluationInstances.forEach(item => {
        if (item.type && arr.indexOf(item.type) === -1) {
          arr.push(item.type)
        }
      })
      return this.evaluationOutputTypeListArr.filter(type => {
        return arr.indexOf(type) !== -1
      })
    },
    haveDataTypeList() {
      return this.evaluationOutputTypeList.filter((item, index) => {
        return this.evaluationFlags[index] === true
      })
    },
    lossList() {
      return this.evaluationInstances.filter(item => {
        return item.type === this.metricTypeMap.loss
      })
    },
    currentbinningData() {
      return this.binningType === 'guest' ? 'data' : 'hostData'
    },
    modelSummaryTitle() {
      return this.modelOutputType === this.modelNameMap.evaluation
        ? 'Evaluation scores' : 'Cross validation scores'
    },
    filterScaleHeader() {
      const isHideCol = this.metricOutputList[0] && this.metricOutputList[0].scaleMethod === 'min_max_scale'
      return isHideCol ? this.scaleHeader.slice(0, 3) : this.scaleHeader
    }
  },
  mounted() {
    // console.log('mounted')
  },
  beforeUpdate() {
    // console.log('before update')
  },
  updated() {
    if (this.modelOutputType === this.modelNameMap.binning && this.modelOutput) {
      if ((this.modelOutput[this.currentbinningData].options && !this.binningSelectValue) || this.lazyBinningType !== this.binningType) {
        this.lazyBinningType = this.binningType
        this.binningSelectValue = this.modelOutput[this.currentbinningData].options[0].value
      }
      const stackBarData = this.modelOutput[this.currentbinningData].stackBarData[this.binningSelectValue]
      const woeData = this.modelOutput[this.currentbinningData].woeData[this.binningSelectValue]
      this.stackBarInstance.setOption(stackBarData, true)
      this.woeInstance.setOption(woeData, true)
    }
    if (this.modelOutputType === this.modelNameMap.oneHot && this.modelOutput) {
      if (this.modelOutput.options && !this.oneHotSelectValue) {
        this.oneHotSelectValue = this.modelOutput.options[0].value
      }
    }
  },
  methods: {
    changeBinning(value) {
      this.binningSelectValue = value
      // console.log(this.binningSelectValue)
      // console.log(this.modelOutput, this.currentbinningData)
      this.stackBarInstance.setOption(this.modelOutput[this.currentbinningData].stackBarData[value], true)
      this.stackBarInstance.setOption(this.modelOutput[this.currentbinningData].woeData[value], true)
    },
    changeOneHot(value) {
      // console.log('changeonehot', value)
    },
    getStackBarInstance(echartInstance) {
      this.stackBarInstance = echartInstance
    },
    getWoeInstance(echartInstance) {
      this.woeInstance = echartInstance
    },
    handlePageChange({ page }) {
      // console.log(page)
    },
    changeCvTab(index) {
      this.$store.dispatch('ChangeCvTab', index)
      const arr = deepCloneArr(this.evaluationFlags)
      if (!arr[index]) {
        arr[index] = true
        this.$store.dispatch('SetCvFlags', arr)
      }
    },
    summarySpanMethod({ row, column, rowIndex, columnIndex }) {
      // console.log(column, rowIndex, columnIndex)
      // if (columnIndex === 0) {
      //   if (rowIndex % 2 === 0) {
      //     return {
      //       rowspan: 2,
      //       colspan: 1
      //     }
      //   }
      //   else {
      //     return {
      //       rowspan: 0,
      //       colspan: 0
      //     }
      //   }
      // }
      // return {
      //   rowspan: 2,
      //   colspan: 1
      // }
    },
    clearEchartInstance() {
      this.echartInstanceList = []
    },
    clickLegend({ curveName, evaluationListIndex, instanceList }) {
      curveName = curveName.replace(/(_tpr|_fpr|_precision|_recall)/g, '')
      let echartInstanceListIndex = -1
      for (let i = 0; i < instanceList.length; i++) {
        const instance = instanceList[i]
        if (instance.legendIndex === evaluationListIndex) {
          echartInstanceListIndex = i
          break
        }
      }
      // console.log(echartInstanceListIndex, curveName)
      if (echartInstanceListIndex === -1) {
        return
      }
      const instance = instanceList[echartInstanceListIndex].instance
      const options = instance.getOption()
      const series = options.series
      // console.log(series)
      for (let i = 0; i < series.length; i++) {
        const item = series[i]
        const name = item.name.replace(/(_tpr|_fpr|_precision|_recall)/g, '')
        if (name === curveName || (!name && curveName === item.pairType)) {
          if (item.itemStyle) {
            item.itemStyle.opacity = item.itemStyle.opacity === 1 ? 0 : 1
          }
          if (item.lineStyle) {
            item.lineStyle.opacity = item.lineStyle.opacity === 1 ? 0 : 1
          }
          if (item.areaStyle) {
            item.areaStyle.opacity = item.areaStyle.opacity === 0.1 ? 0 : 0.1
          }
        }
      }
      instance.setOption(options)
    },
    getEchartInstance(instance, legendIndex) {
      this.echartInstanceList.push({ instance, legendIndex })
    },
    getLossInstance(instance, legendIndex) {
      this.lossInstanceList.push({ instance, legendIndex })
    }
  }
}
</script>

<style lang="scss" scoped>
  .selection-list {
    > li {
      width: 45%;
      height: 300px;
      margin-bottom: 40px;
      overflow-y: auto;
      .echart {
        width: 100%;
        height: 100%;
      }
    }
  }

  .boost-json {
    background: #f8f8fa;
    padding: 20px;
    margin: 10px 20px;
    height: 350px;
    overflow: auto;
  }

  .cv-tab-list {
    margin-bottom: 22px;
    > li {
      margin-right: 20px;
      padding: 5px 0;
      border-bottom: 2px solid transparent;
      font-weight: bold;
      font-size: 16px;
      cursor: pointer;
    }
    .active {
      border-color: #494ece;
      color: #494ece;
    }
  }

  .model-text {
    font-size: 16px;
    font-weight: bold;
    color: #7f7d8e;
    line-height: 24px;
    margin-bottom: 12px;
  }

  .cv-wrapper {
    > li {
      margin-bottom: 15px;
      padding: 24px 32px;
      border: 1px solid #ddd;
      border-radius: 2px;
      .cv-top {
        margin-bottom: 10px;
      }
    }
  }
</style>
