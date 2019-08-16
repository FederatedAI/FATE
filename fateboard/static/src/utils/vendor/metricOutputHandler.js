import store from '@/store/modules/app'
// import { deepClone, filterLineArr } from '@/utils'
import { deepClone } from '@/utils'
import evaluationOptions from '@/utils/chart-options/evaluation-curve'

const { metricTypeMap } = store.state
const curveColor = [
  '#494ECE',
  '#24B68B',
  '#FF8800',
  '#F23BA9',
  '#F34444',
  '#C13CE1',
  '#7C56FF',
  '#3DDBF5',
  '#A7CF02',
  '#FFD503'
]
const curveAlphaColor = [
  '#999CEC',
  '#83E6C9',
  '#FFC37E',
  '#F576C2',
  '#FF8D8D',
  '#DC99EC',
  '#BDAAFF',
  '#A5F2FF',
  '#E1F397',
  '#FFEA84'
]
// const curveColor = ['#f00', '#0f0', '#00f', '#0088ff']
const curveFormatter = (xName, yName, legendData, thresholdsArr = []) => {
  return (params) => {
    // console.log(params)
    let str = ''
    params.forEach((obj, index) => {
      if (!legendData || legendData[index].isActive !== false) {
        let xValue = ''
        if (Array.isArray(thresholdsArr[obj.seriesIndex])) {
          xValue = thresholdsArr[obj.seriesIndex][obj.dataIndex]
        } else {
          xValue = thresholdsArr[obj.dataIndex]
        }
        if (xValue || xValue === 0) {
          str += `${xName}(${obj.seriesName}): ${xValue}<br>`
        }
        const value = Array.isArray(obj.data) ? obj.data[1] : obj.data
        str += `${yName}(${obj.seriesName}): ${value}<br>`
      }
    })
    return str
  }
}
export default function(
  {
    metricOutputList,
    evaluationOutputList,
    data,
    metric_type,
    modelOutputType,
    metric_namespace,
    metric_name,
    meta,
    unit_name,
    name,
    curve_name,
    pair_type,
    modelSummaryData,
    thresholds
  }) {
  let type = ''
  let outputData = ''
  const scaleMethod = meta.method
  curve_name = curve_name && curve_name.replace(/(train_|validate_)/g, '')
  pair_type = pair_type && pair_type.replace(/(train_|validate_)/g, '')
  curve_name = curve_name && curve_name.replace(/(train\.|validate\.)/g, 'fold_')
  thresholds = thresholds || []
  if (metric_type === metricTypeMap.dataIOText) {
    // type = 'text'
    // const arr = []
    // data.forEach(item => {
    //   if (item[0]) {
    //     arr.push(item[0])
    //   }
    // })
    // outputData = `${curve_name}: ${arr.join(', ')}`
  } else if (metric_type === metricTypeMap.sampleText) {
    type = 'text'
    outputData = `${data[0][0]}: ${data[0][1]}`
  } else if (metric_type === metricTypeMap.scale) {
    type = 'text'
    outputData = `method: ${meta.method || 'null'}`
  } else if (metric_type === metricTypeMap.sampleTable) {
    type = 'table'
    const tHeader = [
      {
        prop: 'label',
        label: 'label'
      },
      {
        prop: 'count',
        label: 'count'
      }
    ]
    const tBody = []
    data.forEach(row => {
      tBody.push({
        label: row[0],
        count: row[1]
      })
    })
    const index = { label: 'layer' }
    outputData = {
      tHeader,
      tBody,
      index
    }
  } else if (metric_type === metricTypeMap.intersection) {
    type = 'text'
    data.forEach(item => {
      outputData += `${item[0]}: ${item[1]}` + '<br>'
    })
  } else if (metric_type === metricTypeMap.Accuracy ||
    metric_type === metricTypeMap.Gain ||
    metric_type === metricTypeMap.Lift ||
    metric_type === metricTypeMap.ROC ||
    metric_type === metricTypeMap.RecallMulti ||
    metric_type === metricTypeMap.PrecisionMulti ||
    metric_type === metricTypeMap.loss) {
    const typeArr = Object.keys(metricTypeMap)
    for (let i = 0; i < typeArr.length; i++) {
      if (metric_type === metricTypeMap[typeArr[i]]) {
        if (metric_type === metricTypeMap.RecallMulti ||
          metric_type === metricTypeMap.PrecisionMulti) {
          type = 'Precision Recall'
        } else if (metric_type === metricTypeMap.loss) {
          type = metricTypeMap.loss
        } else {
          type = typeArr[i]
        }
        break
      }
    }
    outputData = deepClone(evaluationOptions)
    outputData.xAxis.name = unit_name
    const seriesObj = {
      name: curve_name,
      type: 'line',
      smooth: false,
      // symbol: 'none',
      symbolSize: 1,
      itemStyle: {
        opacity: 1
      },
      lineStyle: {
        opacity: 1
      },
      data,
      pair_type
    }
    if (metric_type === metricTypeMap.Gain) {
      // outputData.xAxis.name = 'threshold'
      outputData.yAxis.name = 'gain'
      outputData.tooltip.formatter = curveFormatter('Threshold', 'Gain', null, thresholds)
    } else if (metric_type === metricTypeMap.Accuracy) {
      // outputData.xAxis.name = 'threshold'
      outputData.yAxis.name = 'accuracy'
      outputData.tooltip.formatter = curveFormatter('Threshold', 'Accuracy', null, thresholds)
    } else if (metric_type === metricTypeMap.Lift) {
      // outputData.xAxis.name = 'threshold'
      outputData.yAxis.name = 'lift'
      outputData.tooltip.formatter = curveFormatter('Threshold', 'Lift', null, thresholds)
    } else if (metric_type === metricTypeMap.ROC) {
      outputData.yAxis.name = 'tpr'
      seriesObj.areaStyle = {
        color: '#494ece',
        opacity: 0.1
      }

      outputData.tooltip.formatter = (params) => {
        let str = ''
        params.forEach(obj => {
          const xValue = thresholds[obj.dataIndex]
          if (xValue || xValue === 0) {
            str += `Threshold: ${xValue}<br>`
          }
          str += `Tpr(${obj.seriesName}): ${obj.data[1]}<br>`
          str += `Fpr(${obj.seriesName}): ${obj.axisValue}<br>`
        })
        return str
      }
    } else if (metric_type === metricTypeMap.RecallMulti || metric_type === metricTypeMap.PrecisionMulti) {
      // console.log(curve_name,pair_type, metric_namespace)
      outputData.xAxis.name = 'class'
      outputData.xAxis.type = 'category'
      outputData.yAxis.name = 'precision, recall'
      outputData.tooltip.formatter = params => {
        let str = ''
        const xValue = params[0].axisValue
        str += `Class: ${xValue}<br>`
        params.forEach(obj => {
          const value = Array.isArray(obj.data) ? obj.data[1] : obj.data
          str += `${obj.seriesName}: ${value}<br>`
        })
        return str
      }
      const xArr = []
      const valueArr = []
      data.forEach(p => {
        xArr.push(p[0])
        valueArr.push(p[1])
      })
      outputData.xAxis.data = xArr
      seriesObj.data = valueArr
      seriesObj.name = curve_name
      // if (metric_type === metricTypeMap.RecallMulti) {
      //   seriesObj.name += '-recall'
      // } else {
      //   seriesObj.name += '-precision'
      // }
    } else if (metric_type === metricTypeMap.loss) {
      // outputData.xAxis.name = 'iteration'
      outputData.yAxis.name = 'loss'
      outputData.tooltip.formatter = (params) => {
        let str = ''
        const xValue = params[0].axisValue
        str += `iteration: ${xValue}<br>`
        params.forEach(obj => {
          const value = obj.data[1]
          str += `loss(${obj.seriesName}): ${value}<br>`
        })
        return str
      }
    }
    outputData.series.push(seriesObj)
    for (let i = 0; i < evaluationOutputList.length; i++) {
      const item = evaluationOutputList[i]
      if (item.type && item.type === type && item.nameSpace === metric_namespace) {
        if ((metric_type === metricTypeMap.RecallMulti || metric_type === metricTypeMap.PrecisionMulti)) {
          for (let j = 0; j < item.data.series.length; j++) {
            const curve = item.data.series[j]
            let color = ''
            const legendIndex = curve.legendIndex
            if (metric_type === metricTypeMap.PrecisionMulti) {
              color = curveColor[legendIndex]
            } else {
              color = curveAlphaColor[legendIndex]
            }
            const legendObj = { color, text: curve_name }
            if (curve.pair_type && curve.pair_type === pair_type) {
              if (metric_type === metricTypeMap.PrecisionMulti) {
                item.legendData[legendIndex].unshift(legendObj)
              } else {
                item.legendData[legendIndex].push(legendObj)
              }
              seriesObj.itemStyle.color = color
              item.data.series.push(seriesObj)
              return
            }
          }
        }
        const color = curveColor[item.data.series.length] || '#000'
        seriesObj.itemStyle.color = color
        if ((metric_type === metricTypeMap.RecallMulti || metric_type === metricTypeMap.PrecisionMulti)) {
          const legendIndex = item.legendData.length
          let color = ''
          if (metric_type === metricTypeMap.PrecisionMulti) {
            color = curveColor[legendIndex]
          } else {
            color = curveAlphaColor[legendIndex]
          }
          seriesObj.legendIndex = legendIndex
          item.data.series.push(seriesObj)
          const legendObj = { color, text: curve_name }
          item.legendData.push([legendObj])
        } else {
          item.thresholdsArr.push(thresholds)
          if (metric_type === metricTypeMap.Gain) {
            item.data.tooltip.formatter = curveFormatter('Threshold', 'Gain', item.legendData, item.thresholdsArr)
          } else if (metric_type === metricTypeMap.Accuracy) {
            item.data.tooltip.formatter = curveFormatter('Threshold', 'Accuracy', item.legendData, item.thresholdsArr)
          } else if (metric_type === metricTypeMap.Lift) {
            item.data.tooltip.formatter = curveFormatter('Threshold', 'Lift', item.legendData, item.thresholdsArr)
          } else if (metric_type === metricTypeMap.loss) {
            item.data.tooltip.formatter = outputData.tooltip.formatter = (params) => {
              let str = ''
              const xValue = params[0].axisValue
              str += `iteration: ${xValue}<br>`
              params.forEach((obj, index) => {
                if (item.legendData[index].isActive !== false) {
                  const value = obj.data[1]
                  str += `loss(${obj.seriesName}): ${value}<br>`
                }
              })
              return str
            }
          } else if (metric_type === metricTypeMap.RecallMulti || metric_type === metricTypeMap.PrecisionMulti) {
            item.data.tooltip.formatter = params => {
              let str = ''
              const xValue = params[0].axisValue
              str += `Class: ${xValue}<br>`
              params.forEach((obj, index) => {
                if (item.legendData[index].isActive !== false) {
                  const value = Array.isArray(obj.data) ? obj.data[1] : obj.data
                  str += `${obj.seriesName}: ${value}<br>`
                }
              })
              return str
            }
          } else if (metric_type === metricTypeMap.ROC) {
            item.data.tooltip.formatter = (params) => {
              let str = ''
              // console.log(params)
              // console.log(item.legendData)
              params.forEach((obj, index) => {
                if (item.legendData[index].isActive !== false) {
                  const xValue = item.thresholdsArr[index][obj.dataIndex]
                  if (xValue || xValue === 0) {
                    str += `Threshold(${obj.seriesName}): ${xValue}<br>`
                  }
                  str += `Tpr(${obj.seriesName}): ${obj.data[1]}<br>`
                  str += `Fpr(${obj.seriesName}): ${obj.axisValue}<br>`
                }
              })
              return str
            }
          }

          item.data.series.push(seriesObj)
          item.legendData.push({
            color,
            text: curve_name
          })
        }
        return
      }
    }
    let color = ''
    const legendData = []
    if (metric_type === metricTypeMap.RecallMulti || metric_type === metricTypeMap.PrecisionMulti) {
      if (metric_type === metricTypeMap.PrecisionMulti) {
        color = curveColor[0]
      } else {
        color = curveAlphaColor[0]
      }
      outputData.series[0].itemStyle.color = color
      outputData.series[0].legendIndex = 0
      legendData.push([{
        color,
        text: curve_name
      }])
    } else {
      outputData.series[0].itemStyle.color = curveColor[0]
      legendData.push({
        color: curveColor[0],
        text: curve_name
      })
    }

    const echartObj = {
      type,
      nameSpace: metric_namespace,
      data: outputData,
      legendData,
      thresholdsArr: [thresholds]
    }
    if (metric_namespace === 'train') {
      evaluationOutputList.unshift(echartObj)
    } else {
      evaluationOutputList.push(echartObj)
    }
    // }
    return
  } else if (metric_type === metricTypeMap.RecallBinary || metric_type === metricTypeMap.PrecisionBinary) {
    let dataObj = {}
    if (Array.isArray(data)) {
      data.forEach(item => {
        dataObj[item[0]] = item[1]
      })
    } else {
      dataObj = data
    }
    // if (curve_name === 'train_fold_3') {
    //   console.log('data: ', data)
    // }
    const halfObj = {
      name: curve_name,
      type: 'line',
      smooth: false,
      // symbol: 'none',
      symbolSize: 1,
      halfData: dataObj,
      itemStyle: {
        opacity: 1
      },
      lineStyle: {
        opacity: 1
      },
      pair_type
    }

    outputData = deepClone(evaluationOptions)
    outputData.xAxis.name = 'recall'
    outputData.yAxis.name = 'precision'
    outputData.tooltip.formatter = (params) => {
      let str = ''
      // console.log(params)
      params.forEach(obj => {
        const thresholdValue = thresholds[obj.dataIndex]
        if (thresholdValue || thresholdValue === 0) {
          str += `Thresholds(${obj.seriesName}): ${thresholdValue}<br>`
        }
        const value = Array.isArray(obj.data) ? obj.data[1] : obj.data
        str += `Precision(${obj.seriesName}):${value}<br>`
        str += `Recall(${obj.seriesName}):${obj.axisValue}<br>`
      })
      return str
    }
    outputData.series.push(halfObj)
    // console.log('xxxxxxx', pair_type)
    for (let i = 0; i < evaluationOutputList.length; i++) {
      const item = evaluationOutputList[i]
      if (item.type === 'Precision Recall' && item.nameSpace === metric_namespace) {
        for (let j = 0; j < item.data.series.length; j++) {
          const curve = item.data.series[j]
          if (curve.pair_type === pair_type) {
            if (!item.thresholdsArr) {
              item.thresholdsArr = [thresholds]
            } else {
              item.thresholdsArr.push(thresholds)
            }
            // console.log(item.thresholdsArr)
            // if (metric_namespace === 'validate' && curve_name === 'fold_2') {
            //   console.log(thresholds)
            // }
            item.data.tooltip.formatter = (params) => {
              let str = ''
              // console.log(params),
              params.forEach((obj, index) => {
                if (item.legendData[index].isActive !== false) {
                  const thresholdValue = item.thresholdsArr[index][obj.dataIndex]
                  if (thresholdValue || thresholdValue === 0) {
                    str += `Thresholds(${obj.seriesName}): ${thresholdValue}<br>`
                  }
                  const value = Array.isArray(obj.data) ? obj.data[1] : obj.data
                  str += `Precision(${obj.seriesName}):${value}<br>`
                  str += `Recall(${obj.seriesName}):${obj.axisValue}<br>`
                }
              })
              return str
            }
            const prObj = {}
            Object.keys(curve.halfData).forEach(key => {
              const p = []
              if (metric_type === metricTypeMap.RecallBinary) {
                p[0] = dataObj[key]
                p[1] = curve.halfData[key]
              } else {
                p[0] = curve.halfData[key]
                p[1] = dataObj[key]
              }
              if (prObj[p[0]]) {
                if (p[1] > prObj[p[0]][1]) {
                  prObj[p[0]] = p
                }
              } else {
                prObj[p[0]] = p
              }
            })
            // console.log(item.data.series)
            const color = curveColor[item.data.series.length - 1] || '#000'
            item.data.series[j].itemStyle.color = color
            item.legendData.push({
              color,
              text: curve_name
            })
            const sortFn = function(a, b) {
              return a[0] - b[0]
            }
            const curveData = Object.values(prObj)
            const curveDataSort = curveData.sort(sortFn)
            item.data.series[j].data = curveDataSort
            // if (curve_name === 'train_fold_3') {
            //   console.log('curveData: ', curveData)
            //   console.log('curveDataSort: ', curveDataSort)
            // }
            return
          }
        }
        item.data.series.push(halfObj)
        return
      }
    }
    const echartObj = {
      type: 'Precision Recall',
      metric_type,
      nameSpace: metric_namespace,
      data: outputData,
      legendData: []
    }
    if (metric_namespace === 'train') {
      evaluationOutputList.unshift(echartObj)
    } else {
      evaluationOutputList.push(echartObj)
    }
  } else if (metric_type === metricTypeMap['K-S']) {
    const seriesObj = {
      name: curve_name,
      type: 'line',
      smooth: false,
      symbolSize: 1,
      itemStyle: {
        opacity: 1
      },
      lineStyle: {
        opacity: 1
      },
      data,
      pair_type
    }
    type = 'K-S'
    outputData = deepClone(evaluationOptions)
    outputData.xAxis.name = unit_name
    outputData.yAxis.name = 'tpr, fpr'
    outputData.series.push(seriesObj)
    for (let i = 0; i < evaluationOutputList.length; i++) {
      const item = evaluationOutputList[i]
      if (item.type === type && item.nameSpace === metric_namespace) {
        // const curveArr = item.data.series.filter(c => {
        //   return c.pair_type
        // })
        for (let j = 0; j < item.data.series.length; j++) {
          const curve = item.data.series[j]
          if (curve.pair_type && curve.pair_type === pair_type) {
            let maxDValue = 0
            let maxDYValue1 = 0
            let maxDYValue2 = 0
            let maxDXValue = 0
            const legendIndex = curve.legendIndex
            curve.data.forEach((p, pIndex) => {
              const dValue = Math.abs(p[1] - data[pIndex][1])
              if (dValue > maxDValue) {
                maxDValue = dValue
                maxDXValue = p[0]
                maxDYValue1 = p[1]
                maxDYValue2 = data[pIndex][1]
              }
            })
            const formatterObj = /_tpr$/g.test(curve.name)
              ? {
                tpr: curve.name,
                fpr: seriesObj.name,
                pairType: curve.name.replace(/_tpr$/g, '')
              } : {
                tpr: seriesObj.name,
                fpr: curve.name,
                pairType: seriesObj.name.replace(/_tpr$/g, '')
              }
            formatterObj.thresholds = thresholds
            if (item.data.KSFormaterArr) {
              item.data.KSFormaterArr.push(formatterObj)
            } else {
              item.data.KSFormaterArr = [formatterObj]
            }
            item.data.series.push({
              name: '',
              type: 'line',
              symbol: 'none',
              data: [
                [maxDXValue, maxDYValue1],
                [maxDXValue, maxDYValue2]
              ],
              itemStyle: {
                color: curveColor[legendIndex]
              },
              lineStyle: {
                type: 'dashed',
                opacity: 1
              },
              pairType: pair_type
            })
            item.data.tooltip.formatter = params => {
              let str = ''
              // const xAxisName = trimId(params[0].axisId)
              // str += `${xAxisName}: ${params[0].axisValue}<br>`
              item.data.KSFormaterArr.forEach((ksObj, ksIndex) => {
                if (item.legendData[ksIndex].isActive !== false) {
                  // console.log(thresholds, params[0])
                  const thresholdValue = ksObj.thresholds[params[0].dataIndex]
                  if (thresholdValue || thresholdValue === 0) {
                    str += `Threshold: (${ksObj.pairType})${thresholdValue}<br>`
                  }
                  let ksflag = false
                  let v1 = 0
                  let v2 = 0
                  params.forEach(obj => {
                    if (obj.seriesName === ksObj.tpr) {
                      // str += `Tpr(${ksObj.tpr}): ${obj.data[1]}<br>`
                      str += `${ksObj.tpr}: ${obj.data[1]}<br>`
                      v1 = obj.data[1]
                      ksflag = true
                    }
                    if (obj.seriesName === ksObj.fpr) {
                      // str += `Fpr(${ksObj.fpr}): ${obj.data[1]}<br>`
                      str += `${ksObj.fpr}: ${obj.data[1]}<br>`
                      v2 = obj.data[1]
                    }
                  })
                  if (ksflag) {
                    const ks = Math.abs(v1 - v2)
                    str += `KS: ${ks.toFixed(6)}<br>`
                  }
                }
              })
              return str
            }
            let color = ''
            if (/_tpr$/g.test(curve.name)) {
              color = curveAlphaColor[legendIndex]
              seriesObj.itemStyle.color = color
              if (item.data.series.length > j + 1) {
                item.data.series.splice(j + 1, 0, seriesObj)
              } else {
                item.data.series.push(seriesObj)
              }
              item.legendData[legendIndex].push({ color, text: curve_name })
            } else {
              color = curveColor[legendIndex]
              seriesObj.itemStyle.color = color
              item.data.series.splice(j, 0, seriesObj)
              item.legendData[legendIndex].unshift({ color: curveColor[legendIndex], text: curve_name })
            }
            return
          }
        }
        let color = ''
        const legendIndex = item.legendData.length
        seriesObj.legendIndex = legendIndex
        if (/_tpr$/g.test(curve_name)) {
          color = curveColor[legendIndex]
        } else {
          color = curveAlphaColor[legendIndex]
        }
        const arr = [{
          color,
          text: curve_name
        }]
        seriesObj.itemStyle.color = color
        item.data.series.push(seriesObj)
        item.legendData.push(arr)
        return
      }
    }
    let color = ''
    const legendData = []
    if (/_tpr$/g.test(curve_name)) {
      color = curveColor[0]
    } else {
      color = curveAlphaColor[0]
    }
    const arr = [{
      color,
      text: curve_name
    }]
    legendData.push(arr)
    outputData.series[0].itemStyle.color = color
    outputData.series[0].legendIndex = 0
    const echartObj = {
      type,
      nameSpace: metric_namespace,
      data: outputData,
      legendData
    }
    if (metric_namespace === 'train') {
      evaluationOutputList.unshift(echartObj)
    } else {
      evaluationOutputList.push(echartObj)
    }
    return
  } else if (metric_type === metricTypeMap.Summary) {
    // console.log('summary', metric_namespace, data)
    const dataObj = {}
    if (modelSummaryData.tHeader.length === 0) {
      modelSummaryData.tHeader.push({
        prop: 'metric_name',
        label: ''
      })
      modelSummaryData.tHeader.push({
        prop: 'metric_namespace',
        label: 'dataset'
      })
      data.forEach(row => {
        modelSummaryData.tHeader.push({
          prop: row[0],
          label: row[0]
        })
      })
    }
    dataObj.metric_name = metric_name.replace(/(train_|validate_)/g, '')
    dataObj.metric_namespace = metric_namespace
    data.forEach(row => {
      dataObj[row[0]] = row[1]
    })
    modelSummaryData.tBody.push(dataObj)
    modelSummaryData.tBody = modelSummaryData.tBody.sort((a, b) => {
      let r = 0
      const aName = a.metric_name.substr(0, a.metric_name.lastIndexOf('_'))
      const aIndex = a.metric_name.substr(a.metric_name.lastIndexOf('_') + 1)
      const bName = b.metric_name.substr(0, b.metric_name.lastIndexOf('_'))
      const bIndex = b.metric_name.substr(b.metric_name.lastIndexOf('_') + 1)
      if (aName === bName) {
        r = aIndex - bIndex
      } else {
        r = aName.charCodeAt(0) - bName.charCodeAt(0)
      }
      return r
    })
  }
  metricOutputList.push({
    type,
    nameSpace: metric_namespace,
    data: outputData,
    scaleMethod
  })
}
