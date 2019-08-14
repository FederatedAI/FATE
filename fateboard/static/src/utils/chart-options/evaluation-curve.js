// import { trimId } from '@/utils'

export default {
  // color: ['#ff8800', '#f23ba9', '#494ece', '#24b68b', '#A21155'],
  color: ['#f00', '#0f0', '#00f', '#24b68b', '#A21155'],
  backgroundColor: '#fbfbfc',
  tooltip: {
    // enterable: true,
    // position(pos, params, el, elRect, size) {
    //   console.log(pos, params, size)
    //   const obj = { top: 10 }
    //   obj[['left', 'right'][+(pos[0] < size.viewSize[0] / 2)]] = 80
    //   return obj
    // },
    position(pos, params, el, rect, size) {
      const toolTipWidth = el.offsetWidth
      const toolTipHeight = el.offsetHeight
      const viewWidth = size.viewSize[0]
      const viewHeight = size.viewSize[1]
      const leftGap = 20
      const topGap = -10
      let left = pos[0] + leftGap
      let top = pos[1] + topGap
      if (top + toolTipHeight >= viewHeight) {
        top = viewHeight - toolTipHeight
      }
      if (left + toolTipWidth + 5 * leftGap >= viewWidth) {
        left = viewWidth - toolTipWidth - 5 * leftGap
      }
      return { left, top }
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
