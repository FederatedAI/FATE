<template>

  <section>
    <!--metric(进行中状态，只有部分模型有)-->
    <!--{{ metricOutputList.length }}-->
    <ul>
      <li v-for="(output,index) in metricOutputList" :key="index">
        <div v-if="output.type==='line'">
          <echart-container :class="'echart'" :options="output.data"/>
        </div>
        <div v-if="output.type==='KS'">
          <p>{{ output.nameSpace }}</p>
          <echart-container :class="'echart'" :options="output.data"/>
        </div>
      </li>
    </ul>
    <!--最终模型输出-->
    <div v-if="modelOutputType && modelOutput">
      <!--boost-->
      <div v-if="modelOutputType==='HeteroSecureBoost'">
        <pre> {{ modelOutput.formatString }} </pre>
      </div>
      <!--dataio-->
      <div v-else-if="modelOutputType==='DataIO'">
        <el-table
          :data="modelOutput.imputerData"
          highlight-current-row
          fit
          border
          style="margin-bottom: 50px;"
        >
          <el-table-column type="index" label="index" width="100px" align="center"/>
          <el-table-column label="variable" prop="variable" align="center"/>
          <el-table-column label="missing_fill_method" prop="method" align="center"/>
          <el-table-column label="fill_value" prop="value" align="center"/>
        </el-table>

        <el-table
          :data="modelOutput.outlierData"
          highlight-current-row
          fit
          border
        >
          <el-table-column type="index" label="index" width="100px" align="center"/>
          <el-table-column label="variable" prop="variable" align="center"/>
          <el-table-column label="outlier_fill_method" prop="method" align="center"/>
          <el-table-column label="fill_value" prop="value" align="center"/>
        </el-table>
      </div>
      <!--scale-->
      <div v-else-if="modelOutputType==='FeatureScale'">
        <p>method:{{ modelOutput.method }}</p>
        <el-table
          :data="modelOutput.tData"
          highlight-current-row
          fit
          border
        >
          <el-table-column type="index" label="index" width="100px" align="center"/>
          <el-table-column label="variable" prop="variable" align="center"/>
          <el-table-column
            v-for="(item,index) in modelOutput.tHeader"
            :key="index"
            :label="item.label"
            :prop="item.prop"
            align="center"
          />
        </el-table>
      </div>
      <!--lr-->
      <div v-else-if="modelOutputType==='HeteroLR'">
        <el-table
          :data="modelOutput.tData"
          highlight-current-row
          fit
          border
        >
          <el-table-column type="index" label="index" width="100px" align="center"/>
          <el-table-column label="variable" prop="variable" align="center"/>
          <el-table-column label="weight" prop="weight" align="center"/>
        </el-table>
      </div>
      <!--selection-->
      <div v-else-if="modelOutputType==='HeteroFeatureSelection'">
        <ul>
          <li v-for="(item,index) in modelOutput.chartData" :key="index">
            <echart-container :class="'echart'" :options="item"/>
          </li>
        </ul>
      </div>
      <!--intersection-->
      <div v-else-if="modelOutputType==='Intersection'">
        <p>intersection cout: {{ modelOutput.intersection_cout }}</p>
        <p>intersection ratio: {{ modelOutput.intersection_ratio }}</p>
      </div>
      <!--binning-->
      <div v-else-if="modelOutputType==='HeteroFeatureBinning'">
        <el-table
          :data="modelOutput.sourceData"
          highlight-current-row
          fit
          border
        >
          <el-table-column type="index" label="index" width="100px" align="center"/>
          <el-table-column label="variable" prop="variable" align="center"/>
          <el-table-column label="IV" prop="iv" align="center"/>
          <el-table-column label="woe" prop="woe" align="center"/>
          <el-table-column label="monotonicity" prop="monotonicity" align="center"/>
        </el-table>
        <div style="border:1px solid #eee;padding: 30px;margin-top: 25px">
          <el-select v-model="binningSelectValue" @change="changebinning">
            <el-option
              v-for="(item,index) in modelOutput.options"
              :key="index"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-table
            v-if="binningSelectValue"
            :data="modelOutput.variableData[binningSelectValue]"
            highlight-current-row
            fit
            border
            style="margin-top: 20px"
          >
            <el-table-column type="index" label="index" width="100px" align="center"/>
            <el-table-column label="binning" prop="binning" align="center"/>
            <el-table-column label="iv" prop="iv" align="center"/>
            <el-table-column label="woe" prop="woe" align="center"/>
            <el-table-column label="event_count" prop="event_count" align="center"/>
            <el-table-column label="event_ratio" prop="event_ratio" align="center"/>
            <el-table-column label="non_event_count" prop="non_event_count" align="center"/>
            <el-table-column label="non_event_ratio" prop="non_event_ratio" align="center"/>
          </el-table>
        </div>
        <echart-container :class="'echart'" @getEchartInstance="getStackBarInstance"/>
        <echart-container :class="'echart'" @getEchartInstance="getWoeInstance"/>
        <!--{{ modelOutput.stackBarData[binningSelectValue] }}-->
      </div>
      <!--selection-->
      <div v-else-if="modelOutputType==='selection'">
        111
      </div>
      <!--sample-->
      <div v-else-if="modelOutputType==='sample'">
        111
      </div>
      <!--hot-->
      <div v-else-if="modelOutputType==='hot'">
        111
      </div>
      <!--evaluation-->
      <div v-else-if="modelOutputType==='evaluation'">
        111
      </div>
    </div>
  </section>
</template>

<script>
import EchartContainer from '@/components/EchartContainer'

export default {
  name: 'ModelOutput',
  components: {
    EchartContainer
  },
  props: {
    metricOutputList: {
      type: Array,
      default() {
        return []
      }
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
    }
  },
  data() {
    return {
      binningSelectValue: '',
      stackBarInstance: null,
      woeInstance: null
    }
  },
  computed: {},
  mounted() {
  },
  updated() {
    console.log(this.metricOutputList)
    if (this.modelOutputType === 'HeteroFeatureBinning' && this.modelOutput.options && !this.binningSelectValue) {
      this.binningSelectValue = this.modelOutput.options[0].value
      this.stackBarInstance.setOption(this.modelOutput.stackBarData[this.binningSelectValue], true)
      this.woeInstance.setOption(this.modelOutput.woeData[this.binningSelectValue], true)
    }
    if (this.modelOutputType === 'HeteroFeatureBinning') {
      this.stackBarInstance.setOption(this.modelOutput.stackBarData[this.binningSelectValue], true)
      this.woeInstance.setOption(this.modelOutput.woeData[this.binningSelectValue], true)
    }
  },
  methods: {
    changebinning(value) {
      this.stackBarInstance.setOption(this.modelOutput.stackBarData[value], true)
      this.stackBarInstance.setOption(this.modelOutput.woeData[value], true)
    },
    getStackBarInstance(echartInstance) {
      this.stackBarInstance = echartInstance
    },
    getWoeInstance(echartInstance) {
      this.woeInstance = echartInstance
    }

  }
}
</script>

<style scoped>
</style>
