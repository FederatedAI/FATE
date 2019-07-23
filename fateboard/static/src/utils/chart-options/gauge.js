export default {
  tooltip: {
    formatter: '{a} <br/>{b} : {c}%'
  },
  // 图例
  // toolbox: {
  //   feature: {
  //     restore: {},
  //     saveAsImage: {}
  //   }
  // },
  series: [
    {
      title: {
        show: false
      },
      name: '业务指标',
      // 类型：仪表盘
      type: 'gauge',
      // 详情，下方展示数据
      detail: {
        show: true,
        formatter: '{value}%',
        offsetCenter: [0, '30%'],
        fontWeight: 'lighter',
        fontSize: 13
      },
      // 仪表轴线相关样式
      axisLine: {
        lineStyle: {
          // 刻度线分段颜色
          color: [
            [0.3, '#67e0e3'],
            [0.7, '#37a2da'],
            [1, '#0abd00']
          ],
          width: 20
        }
      },
      pointer: {
        show: true
      },
      data: [{ value: 0, name: '完成率' }]
    }
  ]
}
