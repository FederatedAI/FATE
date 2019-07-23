/**
 * 柱形图
 */
export default {
  title: {
    text: ''
  },
  color: ['#8e91df', '#78d0b7'],
  tooltip: {
    trigger: 'axis',
    axisPointer: { // 坐标轴指示器，坐标轴触发有效
      type: 'shadow' // 默认为直线，可选为：'line' | 'shadow'
    }
  },
  legend: {
    show: true,
    right: 0,
    top: '3%',
    orient: 'horizontal',
    itemWidth: 15,
    itemHeight: 15
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  yAxis: {
    type: 'value'
    // axisTick: {
    //   show: false
    // },
    // axisLine: {
    //   show: false
    // },
    // axisLabel: {
    //   show: false
    // },
    // splitLine: {
    //   show: false
    // }
  },
  xAxis: {
    type: 'category',
    // data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    data: []
    // axisTick: {
    //   show: false
    // },
    // axisLine: {
    //   show: false
    // },
    // axisLabel: {
    //   show: false
    // },
    // splitLine: {
    //   show: false
    // }
  },
  series: [
    // {
    //   name: '',
    //   type: 'bar',
    //   data: [],
    //   stack: null
    //   // barWidth: '20%',
    //   // data: [0.05, 0.16, 0.02, 0.24, 0.31, 0.45, 0.75],
    //   // stack: 'event'
    // },
    // {
    //   name: '',
    //   type: 'bar',
    //   data: [],
    //   stack: null
    //   // barWidth: '20%',
    //   // data: [0.15, 0.30, 0.42, 0.54, 0.61, 0.15, 0.05],
    //   // stack: 'event'
    // }
  ]
}
