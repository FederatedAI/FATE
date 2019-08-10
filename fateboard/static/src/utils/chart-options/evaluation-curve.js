// import { trimId } from '@/utils'

export default {
  // color: ['#ff8800', '#f23ba9', '#494ece', '#24b68b', '#A21155'],
  color: ['#f00', '#0f0', '#00f', '#24b68b', '#A21155'],
  backgroundColor: '#fbfbfc',
  tooltip: {
    // enterable: true,
    position(pos, params, el, elRect, size) {
      const obj = { top: 10 }
      obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 80
      return obj
    },
    axisPointer: {
      type: 'line'
    },
    trigger: 'axis'
  },
  // legend: {
  //   show: true,
  //   right: '5%',
  //   top: '3%',
  //   orient: 'horizontal',
  //   itemWidth: 15,
  //   itemHeight: 15
  // },
  itemStyle: {},
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
    // nameTextStyle: {
    //   color: 'transparent'
    // },
    nameCap: 10
  },
  yAxis: {
    type: 'value',
    name: ''
  },
  series: [
    // {
    //   name: 'value',
    //   type: 'line',
    //   smooth: true,
    //   symbol: 'none',
    //   // areaStyle: {
    //   //   color: '#3398DB',
    //   //   opacity: 0.5
    //   // },
    //   data: []
    // }
  ]
}
