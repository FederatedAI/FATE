// const data = [0.15, 0.30, 0.42, 0.54, 0.61, 0.65, 0.85]
const series = [
  {
    // name: 'background',
    type: 'bar',
    barWidth: '20%',
    barGap: '-100%',
    itemStyle: {
      color: '#e8e8ef'
    },
    // data: [1, 1, 1, 1, 1, 1, 1],
    data: [],
    label: {
      show: true,
      position: 'right',
      // formatter(params) {
      //   return data[params.dataIndex]
      // }
      color: '#999',
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
  color: ['#878ada'],
  title: {
    text: '',
    textStyle: {
      color: '#606266',
      fontSize: 16,
      fontFamily: '"Lato", "proxima-nova", "Helvetica Neue", Arial, sans-serif'
    }
  },
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  grid: {
    left: '3%',
    right: '10%',
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
