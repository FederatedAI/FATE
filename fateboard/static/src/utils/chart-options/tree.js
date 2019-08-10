export default {
  tooltip: {
    trigger: 'item',
    triggerOn: 'mousemove'
  },
  series: [
    {
      type: 'tree',

      data: [
        {
          name: 'test',
          children: [
            {
              name: 'child1',
              children: [
                {
                  name: 'child1-1'
                },
                {
                  name: 'child1-2'
                }
              ]
            },
            {
              name: 'child2',
              children: [
                {
                  name: 'child2-1'
                },
                {
                  name: 'child2-2'
                },
                {
                  name: 'child2-3'
                }
              ]
            }
          ]
        }
      ],

      left: '2%',
      right: '2%',
      top: '8%',
      bottom: '20%',

      symbol: 'emptyCircle',
      symbolSize: [100, 25],

      orient: 'vertical',

      expandAndCollapse: true,

      label: {
        normal: {
          position: 'inside',
          // rotate: -90,
          verticalAlign: 'middle',
          // align: 'right',
          fontSize: 12
        }
      },
      itemStyle: {},
      leaves: {
        label: {
          // normal: {
          //   position: 'bottom',
          //   rotate: -90,
          //   verticalAlign: 'middle',
          //   align: 'left'
          // }
        }
      },

      animationDurationUpdate: 750
    }
  ]
}
