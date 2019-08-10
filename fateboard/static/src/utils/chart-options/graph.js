export default {
  tooltip: {
    show: false
  },
  animationDurationUpdate: 500,
  animationEasingUpdate: 'quinticInOut',
  grid: {
    top: '20%',
    bottom: '20%',
    left: '10%',
    right: '10%'
  },
  series: [
    {
      type: 'graph',
      layout: 'none',
      roam: false,
      label: {
        normal: {
          show: true,
          offset: [0, 25],
          fontSize: 14
        }
        // show: true,
        // color: '#333',
        // borderWidth: 1,
        // borderRadius: 4,
        // borderColor: '#333',
        // // backgroundColor: '#fff',
        // padding: [10, 30],
        // lineHeight: 20
      },
      // rect,circle,roundRect,triangle,diamond,pin,arrow,none
      symbol: 'circle',
      symbolSize: [30, 30],
      symbolOffset: [0, 0],
      edgeSymbol: ['none', 'arrow'],
      edgeSymbolSize: [3, 8],
      // edgeLabel: {
      //   show: true,
      //   textStyle: {
      //     fontSize: 20
      //   }
      // },
      data: [],
      links: [],
      itemStyle: {
        // color: 'transparent'
      },
      lineStyle: {
        normal: {
          color: '#7f7d8e',
          opacity: 0.9,
          width: 1,
          curveness: 0
        }
      }
    }
  ]
}
