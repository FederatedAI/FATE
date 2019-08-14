/**
 * 处理graph树形关系图中间件，计算节点坐标
 * @param data 后台给出的节点名（id）以及依赖关系
 * @returns {{dataList: Array, linksList: Array}} 节点坐标数据；节点连接数据
 */

export default function(data) {
  const faiedColor = '#ff6464'
  const successColor = '#24b68b'
  const runningColor = '#494ece'
  // const dependencies = data['dependencies']
  // const list = data['component_list']
  const { dependencies, component_list: list, component_module: modelNameList } = data
  let variableLevel = 0 // 初始level（根节点）为0，当父节点的level大于当前level,自增1
  let variableIndex = 0 // 初始索引，每一个level中第几个index计数，当level增加后，index恢复为1
  const unit = 10 // 单位，可任意指定数值
  const levelMap = [] // 同级中有几个Node,再根据最大值计算坐标
  const dataList = []
  const linksList = []
  // dataList，后台数据只有一个name以及依赖关系，绘图坐标需要计算属性，故新声明X和Y，根据节点进程，遍历List，根据依赖关系计算坐标
  // dataList属性包括{
  // name:'node1', 节点名字
  // level:0, 节点属于哪个层级，可算出y坐标
  // index:1 节点在该层索引，从1开始，可算出
  const nameList = []
  list.forEach(item => {
    nameList.push(item.component_name)
  })

  // 遍历组件List
  for (let i = 0; i < list.length; i++) {
    // 获取list中对应dependencies
    const linkArr = dependencies[list[i].component_name]
    let color = '#bbbbc8'
    if (list[i].status === 'failed') {
      color = faiedColor
    } else if (list[i].status === 'running') {
      color = runningColor
    } else if (list[i].status === 'success') {
      color = successColor
    }
    // 2.有父节点，分情况判断
    if (linkArr) {
      // 遍历每个节点父节点的dependencies数组
      for (let j = 0; j < linkArr.length; j++) {
        // 第一个linkArr不存在没有父节点，例如五个节点如果只有一条线，则只有四个Link线
        // 将该节点置为target，父节点置为source，制作图的导向线
        linksList.push({
          target: i,
          source: nameList.indexOf(linkArr[j])
        })
      }

      // 3. 找层级最大的父节点
      let parentNode = null
      dataList.forEach(item => {
        for (let i = 0; i < linkArr.length; i++) {
          if (item.name === linkArr[i]) {
            if (parentNode) {
              if (item.level < parentNode.level) {
                parentNode = item
              }
            } else {
              parentNode = item
            }
          }
        }
      })
      // 4. 若当前计数level小于或等于父节点level，level自增1，将该层索引插入levelMap作为该层节点数
      // console.log(variableLevel, parentNode)
      if (parentNode && variableLevel <= parentNode.level) {
        ++variableLevel
        levelMap.push(variableIndex)
        variableIndex = 1
      } else {
        // 5. 当前计数level大于父节点level，level不变，索引自增1
        ++variableIndex
      }
      // 6. 遍历到最后一个节点，需把当前index推到levelMap
      if (i === list.length - 1) {
        levelMap.push(variableIndex)
      }
    } else {
      // 1. 没有父节点，level不增加 索引自增1
      ++variableIndex
    }
    const label = {
      color,
      borderColor: color
    }
    const itemStyle = {
      color
    }
    dataList.push({
      name: list[i].component_name,
      componentType: modelNameList[list[i].component_name],
      level: variableLevel,
      // index: variableIndex,
      index: variableIndex,
      sourceLabel: label,
      label,
      itemStyle
    })
  }

  // console.log('dataList:', dataList)
  // console.log('levelMap', levelMap)
  // 最大的同一级中的节点数量
  const maxLevelCount = Math.max(...levelMap)
  // x轴总长度
  const xLength = (maxLevelCount - 1) * unit
  // 插入节点data坐标

  // for (let i = 1; i < dataList.length; i++) {
  //   if (dataList[i].level === dataList[i - 1].level) {
  //     for (let j = i; j < dataList.length; j++) {
  //       dataList[j].level = dataList[j].level + 1
  //     }
  //   }
  // }
  const variableY = 0
  dataList.map((node, index) => {
    // 7.获取节点该层中节点数量
    const levelNodesCount = levelMap[node.level]

    // 8. 计算x坐标，若该层节点数为最大节点数，直接乘以单位
    let x = 0
    if (levelNodesCount === maxLevelCount) {
      x = (node.index - 1) * unit * 3
      // 若不为节点数，则用最大长度除以一个节点所占宽度，再乘以索引计算X坐标
    } else {
      x = xLength / (levelNodesCount + 1) * node.index * 3
    }
    node.x = x || 0
    // 层级决定y坐标
    // for (let i = index; i >= 0; i--) {
    //   if (dataList[i].level === node.level) {
    //     ++variableY
    //     break
    //   }
    // }
    node.y = (node.level + variableY) * unit
  })

  // for (let i = 1; i < dataList.length; i++) {
  //   if (dataList[i].level === dataList[i - 1].level) {
  //     dataList[i].symbolSize = [120, 20]
  //     dataList[i - 1].symbolSize = [120, 20]
  //   }
  // }
  console.log(dataList, linksList)
  return { dataList, linksList }
}
