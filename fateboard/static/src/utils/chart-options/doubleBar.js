/**
 * 柱形图
 */
// const data = [0.15, 0.30, 0.42, 0.54, 0.61, 0.65, 0.85]
const series = [
  {
    // name: 'background',
    type: 'bar',
    barWidth: '20%',
    barGap: '-100%',
    itemStyle: {
      color: '#aaa'
    },
    // data: [1, 1, 1, 1, 1, 1, 1],
    data: [],
    label: {
      show: true,
      position: 'right',
      // formatter(params) {
      //   return data[params.dataIndex]
      // }
      formatter: null
    },
    tooltip: {
      show: false
    }
  },
  {
    name: 'value',
    type: 'bar',
    barWidth: '20%',
    data: []
  }
]
const options = {
  // color: ['#3398DB', '#D22123', '#20D252', '#1022F0', '#A21155'],
  title: {
    text: ''
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: { // 坐标轴指示器，坐标轴触发有效
      type: 'shadow' // 默认为直线，可选为：'line' | 'shadow'
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  yAxis: {
    type: 'category',
    // data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    data: [],
    axisTick: {
      show: false
    },
    axisLine: {
      show: false
    },
    // axisLabel: {
    //   show: false
    // },
    splitLine: {
      show: false
    }
  },
  xAxis: {
    type: 'value',
    axisTick: {
      show: false
    },
    axisLine: {
      show: false
    },
    axisLabel: {
      show: false
    },
    splitLine: {
      show: false
    }
  },
  series
}

export default options
