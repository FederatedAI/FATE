import store from '@/store/modules/app'
import { deepClone } from '@/utils'

import stackBarOptions from '@/utils/chart-options/stackBar'
import doubleBarOptions from '@/utils/chart-options/doubleBar'

const { modelNameMap } = store.state
const handleBinningData = data => {
  if (data && Object.keys(data).length > 0) {
    const sourceData = []
    const options = []
    const variableData = {}
    const stackBarData = {}
    const woeData = {}
    Object.keys(data).forEach(key => {
      const tableData = []
      let min = 0
      const formatterArr = []
      const iterationArr = data[key].ivArray.length > 0 ? data[key].ivArray : data[key].splitPoints
      iterationArr.forEach((item, index, self) => {
        const point = data[key].splitPoints[index] || data[key].splitPoints[index - 1]
        let binning = ''
        let formatterBinning = ''
        if (point) {
          if (min === 0) {
            binning = `${key} < ${point}`
            formatterBinning = `(-∞,${point})`
          } else if (index === self.length - 1) {
            binning = `${key} >= ${point}`
            formatterBinning = `[${point},+∞)`
          } else {
            binning = `${min} <= ${key} < ${point}`
            formatterBinning = `[${min},${point})`
          }
          min = point
        }
        tableData.push({
          binning,
          event_count: data[key].eventCountArray[index],
          event_ratio: data[key].eventRateArray[index],
          non_event_count: data[key].nonEventCountArray[index],
          non_event_ratio: data[key].nonEventRateArray[index],
          woe: data[key].woeArray[index],
          iv: data[key].ivArray[index]
        })
        formatterArr.push({
          formatterBinning,
          event_count: data[key].eventCountArray[index],
          event_ratio: data[key].eventRateArray[index],
          non_event_count: data[key].nonEventCountArray[index],
          non_event_ratio: data[key].nonEventRateArray[index],
          woe: data[key].woeArray[index]
        })
      })
      variableData[key] = tableData
      const eventOptions = deepClone(stackBarOptions)
      const woeOptions = deepClone(stackBarOptions)
      eventOptions.tooltip.formatter = (params) => {
        const obj = formatterArr[params[0].dataIndex]
        return `${obj.formatterBinning}<br>Event Count: ${obj.event_count}<br>
                Event Ratio: ${obj.event_ratio}<br>Non-Event Count: ${obj.non_event_count}<br>
                Non-Event Ratio: ${obj.non_event_ratio}<br>`
      }
      woeOptions.tooltip.trigger = 'item'
      woeOptions.tooltip.formatter = (params) => {
        const obj = formatterArr[params.dataIndex]
        return `${obj.formatterBinning}<br>Woe: ${obj.woe}<br>`
      }
      eventOptions.series.push({
        name: 'event count',
        type: 'bar',
        data: data[key].eventCountArray,
        stack: 'event'
        // barWidth: '20%',
      })

      eventOptions.series.push({
        name: 'non-event count',
        type: 'bar',
        data: data[key].nonEventCountArray,
        stack: 'event'
        // barWidth: '20%',
      })
      for (let i = 1; i <= data[key].eventCountArray.length; i++) {
        eventOptions.xAxis.data.push(i)
        woeOptions.xAxis.data.push(i)
      }
      stackBarData[key] = eventOptions

      woeOptions.series.push({
        name: 'woe',
        type: 'bar',
        data: data[key].woeArray
        // barWidth: '20%',
      })
      woeOptions.series.push({
        // name: 'woe ',
        type: 'line',
        tooltip: {
          show: false
        },
        data: data[key].woeArray
        // barWidth: '20%',
      })
      woeData[key] = woeOptions
      sourceData.push({
        variable: key,
        iv: data[key].iv,
        // woe: data[key].woe,
        monotonicity: data[key].isWoeMonotonic ? data[key].isWoeMonotonic.toString() : 'false'
      })
      options.push({
        value: key,
        label: key
      })
    })
    return {
      sourceData,
      options,
      variableData,
      stackBarData,
      woeData
    }
  } else {
    return null
  }
}
export default function({ outputType, responseData }) {
  let output = {
    isNoModelOutput: false
  }
  if (!responseData || Object.keys(responseData).length === 0) {
    output.isNoModelOutput = true
    return output
  }
  if (outputType === modelNameMap.boost) {
    if (Object.keys(responseData).length > 0) {
      output.formatObj = responseData
      output.formatString = JSON.stringify(responseData, null, 2)
    }
    // dataio
  } else if (outputType === modelNameMap.dataIO) {
    const imputerData = []
    const outlierData = []
    const { imputerParam, outlierParam } = responseData
    const isExistImputerParams = imputerParam && imputerParam.missingReplaceValue && Object.keys(imputerParam.missingReplaceValue).length > 0
    const isExistOutlierParams = outlierParam && outlierParam.outlierReplaceValue && Object.keys(outlierParam.outlierReplaceValue).length > 0
    if (isExistImputerParams) {
      Object.keys(imputerParam.missingReplaceValue).forEach(key => {
        imputerData.push({
          variable: key,
          ratio: imputerParam.missingValueRatio[key],
          value: imputerParam.missingReplaceValue[key]
        })
      })
      output.imputerData = imputerData
    }
    if (isExistOutlierParams) {
      Object.keys(outlierParam.outlierReplaceValue).forEach(key => {
        outlierData.push({
          variable: key,
          ratio: outlierParam.outlierValueRatio[key],
          value: outlierParam.outlierReplaceValue[key]
        })
      })
      output.outlierData = outlierData
    }
    if (!isExistOutlierParams && !isExistOutlierParams) {
      output.isNoModelOutput = true
    }
    // console.log(output)
  } else if (outputType === modelNameMap.intersection) {
    if (responseData) {
      output = responseData
    } else {
      output.isNoModelOutput = true
    }
  } else if (outputType === modelNameMap.scale) {
    const data = responseData && responseData.colScaleParam
    const header = responseData && responseData.header
    const tBody = []
    if (data && header) {
      // Object.keys(data).forEach(key => {
      //   const obj = data[key]
      //   obj.variable = key
      //   tBody.push(obj)
      // })
      header.forEach(head => {
        const row = data[head]
        if (row) {
          row.variable = head
          tBody.push(row)
        }
      })
      output = {
        tBody
      }
    } else {
      output.isNoModelOutput = true
    }
  } else if (outputType === modelNameMap.homoLR || outputType === modelNameMap.heteroLR) {
    const { weight, intercept, isConverged, iters, needOneVsRest } = responseData
    const tData = []
    if (weight && Object.keys(weight).length > 0) {
      Object.keys(weight).forEach(key => {
        tData.push({
          variable: key,
          weight: weight[key]
        })
      })
      if (!needOneVsRest) {
        tData.push({
          variable: 'intercept',
          weight: intercept
        })
      }
      if (needOneVsRest) {
        output = {
          tData
        }
      } else {
        output = {
          tData,
          isConverged,
          iters
        }
      }
    } else {
      output.isNoModelOutput = true
    }
  } else if (outputType === modelNameMap.selection) {
    const data = responseData && responseData.results
    if (data) {
      const chartData = []
      data.forEach(item => {
        const { filterName, featureValues, leftCols } = item
        const leftObj = leftCols.leftCols
        if (filterName && featureValues && Object.keys(featureValues).length > 0) {
          const options = deepClone(doubleBarOptions)
          options.title.text = filterName
          const sortArr = []
          Object.keys(featureValues).forEach(key => {
            sortArr.push({ key, value: featureValues[key], isLeft: leftObj[key] })
          })
          const sortKeyArr = []
          const sortValueArr = []
          sortArr.sort((a, b) => {
            return a.value - b.value
          })
          sortArr.forEach(item => {
            sortKeyArr.push(item.key)
            const valueObj = {
              value: item.value
            }
            if (!item.isLeft) {
              valueObj.itemStyle = { color: '#999' }
            }
            sortValueArr.push(valueObj)
          })
          options.yAxis.data = sortKeyArr
          const value = []
          sortValueArr.forEach(item => {
            value.push(item.value)
          })
          options.series[0].label.formatter = function(params) {
            return value[params.dataIndex]
          }
          let max = 0
          value.forEach((item, index, arr) => {
            if (item > max) {
              max = item
            }
            arr[index] = item
          })
          Object.keys(featureValues).forEach(() => {
            options.series[0].data.push(max * 1.2)
          })
          options.series[1].data = sortValueArr
          options.containerHeight = value.length * 20 + 150
          chartData.push(options)
          // console.log(sortArr)
        } else {
          output.isNoModelOutput = true
        }
      })
      output = { chartData }
    } else {
      output.isNoModelOutput = true
    }
  } else if (outputType === modelNameMap.oneHot) {
    const data = responseData && responseData.colMap
    const options = []
    const variableData = {}
    if (data) {
      Object.keys(data).forEach(key => {
        options.push({
          value: key,
          label: key
        })
        variableData[key] = []
        data[key].encodedVariables.forEach((item, index) => {
          variableData[key].push({
            encoded_vector: item,
            value: data[key].values[index]
          })
        })
      })
      output = {
        options,
        variableData
      }
    } else {
      output.isNoModelOutput = true
    }
  } else if (outputType === modelNameMap.binning) {
    const data = responseData && responseData.binningResult && responseData.binningResult.binningResult
    const hostData = responseData && responseData.hostResults && responseData.hostResults.host && responseData.hostResults.host.binningResult

    if ((data && Object.keys(data).length > 0) || (hostData && Object.keys(hostData).length > 0)) {
      output.data = handleBinningData(data)
      output.hostData = handleBinningData(hostData)
    } else {
      output.isNoModelOutput = true
    }
  }
  return output
}
