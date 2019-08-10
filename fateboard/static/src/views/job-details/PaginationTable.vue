<template>
  <div>
    <el-table
      :data="tablePageData"
      highlight-current-row
      fit
      border
      empty-text="No data"
      element-loading-text="Loading"
      style="margin-bottom: 20px;"
    >
      <el-table-column v-if="haveIndex" prop="tablePageIndex" label="index" width="100" align="center"/>
      <el-table-column
        v-for="(item,index) in header"
        :key="index"
        :label="item.label"
        :prop="item.prop"
        align="center"
        show-overflow-tooltip
      />
    </el-table>
    <div class="flex flex-end">
      <el-pagination
        :total="tableData.length"
        :current-page.sync="currentPage"
        :page-size="pageSize"
        background
        layout="prev, pager, next"
        @current-change="changePage"
      />
    </div>
  </div>
</template>

<script>
import Pagination from '@/components/Pagination'

export default {
  name: 'DataOutput',
  components: {
    Pagination
  },
  props: {
    header: {
      type: Array,
      default() {
        return []
      }
    },
    tableData: {
      type: Array,
      default() {
        return []
      }
    },
    pageSize: {
      type: Number,
      default: 10
    },
    haveIndex: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      currentPage: 1
    }
  },
  computed: {
    tablePageData() {
      const data = []
      for (let i = 0; i < this.tableData.length; i++) {
        const row = this.tableData[i]
        const limitPre = i >= (this.currentPage - 1) * this.pageSize
        const limitNext = i < this.currentPage * this.pageSize
        // console.log(i, limitPre, limitNext)
        if (limitPre && limitNext) {
          if (this.haveIndex) {
            row.tablePageIndex = i + 1
          }
          data.push(row)
        }
      }
      return data
    }
  },
  mounted() {
  },
  methods: {
    changePage(page) {
      this.currentPage = page
    },
    sliceArray(arr) {
      let index = 0
      const newArr = []
      while (index < arr.length) {
        newArr.push(arr.slice(index, index += this.pageSize))
      }
      return newArr
    }
  }
}
</script>

<style lang="scss">
</style>
