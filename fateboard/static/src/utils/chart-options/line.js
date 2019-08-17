export default {
  color: ['#3398DB', '#D22123', '#20D252', '#1022F0', '#A21155'],
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'line'
    }
  },
  grid: {
    left: '3%',
    right: '10%',
    bottom: '3%',
    containLabel: true
  },
  xAxis: {
    type: 'value',
    axisTick: {
      alignWithLabel: true
    },
    name: '',
    nameLocation: 'center',
    nameCap: 10
  },
  yAxis: [{
    type: 'value'
  }],
  series: []
}
