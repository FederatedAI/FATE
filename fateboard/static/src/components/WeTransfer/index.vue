<template>
  <div class="my-transfer">
    <el-dialog :title="title" :visible.sync="isShow" width="640px" @open="initData" @opened="fieldDrop">
      <div class="d-flex align-items-center justify-content-center">
        <el-transfer
          v-model="showFields"
          :titles="['隐藏字段', '显示字段']"
          :format="{noChecked: '${total}',hasChecked: '${checked}/${total}'}"
          :data="columnsData"
          :filter-method="filterMethod"
          :props="fieldAlias"
          filter-placeholder="搜索"
          filterable
          target-order="push"
          style="text-align: left; display: inline-block"
          @right-check-change="rightCheckChange"
        />
        <div class="d-flex flex-column ml-2">
          <el-button
            :disabled="upDisabled"
            type="primary"
            icon="el-icon-arrow-up"
            size="medium"
            class="ml-0 el-transfer__button"
            circle
            @click="upCheckedField"
          />
          <el-button
            :disabled="downDisabled"
            type="primary"
            icon="el-icon-arrow-down"
            size="medium"
            class="mt-1 ml-0 el-transfer__button"
            circle
            @click="downCheckedField"
          />
        </div>
      </div>
      <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="done">确定</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script>
import Sortable from 'sortablejs'
import { simpleDeepClone } from '../../utils'
export default {
  name: 'WeTransfer',
  props: {
    // 对话框标题
    title: {
      default: function() {
        return '编辑表格显示字段'
      },
      type: String
    },
    fixFirst: { // 显示项固定首行
      default: function() {
        return false
      },
      type: Boolean
    },
    // 数据源别名字段
    fieldAlias: {
      default: function() {
        return { key: 'key', label: 'label', disabled: 'disabled' }
      },
      type: Object
    },

    // 表格字段
    columnsData: {
      default: function() {
        return []
      },
      type: Array,
      required: true
    },
    doneFun: {
      default: function() {
        return () => {}
      },
      type: Function
    }
  },
  data() {
    return {
      filterMethod(query, item) {
        return item.label.indexOf(query) > -1
      },
      showFields: [],
      sortable: null,
      isShow: false,
      checkFields: [] // 右侧选中字段数组

    }
  },
  computed: {
    upDisabled: {
      get() {
        if (this.checkFields.length !== 1) {
          return true
        } else if (this.fixFirst && this.showFields.indexOf(this.checkFields[0]) === 1) {
          return true
        } else if (!this.fixFirst && this.showFields.indexOf(this.checkFields[0]) === 0) {
          return true
        }
        return false
      }
    },
    downDisabled: {
      get() {
        if (this.checkFields.length !== 1) {
          return true
        } else if (this.showFields.indexOf(this.checkFields[0]) === this.showFields.length - 1) {
          return true
        }
        return false
      }
    }
  },

  methods: {
    // 显示穿梭框
    show() {
      this.isShow = true
    },
    // 点击确定返回数据
    done() {
      this.isShow = false
      const showColumns = []
      const tmpData = simpleDeepClone(this.columnsData)
      this.showFields.forEach((v) => {
        for (let i = 0; i < tmpData.length; i++) {
          const item = tmpData[i]
          if (item.prop === v) {
            item.isShow = true
            showColumns.push(item)
            tmpData.splice(i, 1)
            i--
          } else {
            item.isShow = false
          }
        }
      })
      if (showColumns.length === 0) {
        for (const v of tmpData) {
          v.isShow = false
        }
      }
      const newColumns = showColumns.concat(tmpData)
      this.doneFun(newColumns)
    },
    // 初始化数据
    initData() {
      this.showFields = []
      const key = this.fieldAlias.key || 'key'
      if (this.fixFirst) {
        this.columnsData[0].disabled = true
      }
      this.columnsData.forEach((v) => {
        if (v.isShow) {
          this.showFields.push(v[key])
        }
      })
    },
    // 字段拖动
    fieldDrop() {
      const panel = document.querySelectorAll('.el-transfer-panel__list')[1]
      this.sortable = Sortable.create(panel, {
        animation: 180,
        draggable: this.fixFirst ? 'label:not(:first-of-type)' : 'label',
        delay: 0,
        onEnd: evt => {
          const oldItem = this.showFields[evt.oldIndex]
          this.showFields.splice(evt.oldIndex, 1)
          this.showFields.splice(evt.newIndex, 0, oldItem)
        }
      })
    },
    // 右侧选中元素有变动时触发
    rightCheckChange(value) {
      this.checkFields = value
      console.log(this.checkFields)
    },
    // 上移
    upCheckedField() {
      const field = this.checkFields[0]
      const index = this.showFields.indexOf(field)
      this.showFields.splice(index, 1)
      this.showFields.splice(index - 1, 0, field)
    },
    // 下移
    downCheckedField() {
      const field = this.checkFields[0]
      const index = this.showFields.indexOf(field)
      this.showFields.splice(index, 1)
      this.showFields.splice(index + 1, 0, field)
    }
  }
}
</script>
 <style lang="scss">
 .my-transfer{
   .el-transfer-panel{
     .el-transfer-panel__body{
       height: 350px;
     }
     .el-checkbox-group{
       height: 100%;
     }
      .el-transfer-panel__header .el-checkbox__label{
        font-size:14px;
      }
      .el-transfer-panel__body .el-checkbox__label{
        font-size:12px;
      }
   }

 }
 </style>
