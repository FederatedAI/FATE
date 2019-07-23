<template>
  <div ref="myEchart" :class="className" :id="id"/>
</template>
<script>
import echarts from 'echarts'

export default {
  props: {
    className: {
      type: String,
      default: ''
    },
    id: {
      type: String,
      default: ''
    },
    options: {
      type: Object,
      default() {
        return {}
      }
    }
  },
  data() {
    return {
      echarts,
      echartInstance: null
    }
  },
  mounted() {
    this.initChart()
  },
  beforeDestroy() {
    if (!this.echartInstance) {
      return
    }
    this.echartInstance.dispose()
    this.echartInstance = null
    window.removeEventListener('resize', this.resize)
  },
  methods: {
    initChart() {
      this.echartInstance = this.echarts.init(this.$refs.myEchart)
      window.addEventListener('resize', this.resize)
      this.$emit('getEchartInstance', this.echartInstance)
      this.$emit('getEchart', this.echarts)
      // 把配置和数据放这里
      this.echartInstance.setOption(this.options)
    },
    resize() {
      this.echartInstance.resize()
    }
  }
}
</script>

<style>
  .default-echart {
    width: 75vw;
    height: 75vh;
  }
</style>
