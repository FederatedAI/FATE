/**
 * 折线图
 */
export default {
  color: ['#3398DB', '#D22123', '#20D252', '#1022F0', '#A21155'],
  tooltip: {
    trigger: 'axis',
    axisPointer: { // 坐标轴指示器，坐标轴触发有效
      type: 'shadow' // 默认为直线，可选为：'line' | 'shadow'
    }
  },
  grid: {
    left: '3%',
    right: '10%',
    bottom: '3%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    data: [],
    axisTick: {
      alignWithLabel: true
    },
    name: '',
    nameLocation: 'end',
    nameCap: 10
  },
  yAxis: [{
    type: 'value'
  }],
  series: [
    {
      name: 'value',
      type: 'line',
      // 线转折变平滑
      // smooth: true,
      // 去掉点
      symbol: 'none',
      // 面积
      // areaStyle: {
      //   color: '#3398DB',
      //   opacity: 0.5
      // },
      data: []
    }
  ]
}
