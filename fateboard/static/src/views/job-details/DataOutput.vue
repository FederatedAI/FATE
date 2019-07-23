<template>
  <section>
    <el-table
      :data="tBody"
      highlight-current-row
      border
      size="mini"
      height="450"
      style="width: 100%;"
    >
      <el-table-column type="index" label="index" width="134" align="center"/>
      <el-table-column
        v-for="(item,index) in header[page-1]"
        :key="index"
        :label="item.label"
        :prop="item.prop"
        width="134"
        align="center"/>
    </el-table>
    <div class="flex flex-end">
      <div v-if="total>0" class="pagination flex flex-center">
        <span>Total: {{ total }}</span>
        <i class="el-icon-arrow-left icon-arrow pointer" @click="changePage('minus')"/>
        <div class="flex flex-center">
          <span
            v-if="page-1>=4"
            :class="{'page-count-active':page===1}"
            class="page-count pointer"
            @click="page=1">1</span>
          <span v-if="page-1>=4">...</span>
          <span
            v-for="(item,index) in totalArray"
            v-if="Math.abs(page-item)<=2"
            :key="index"
            :class="{'page-count-active':page===item}"
            class="page-count pointer"
            @click="page=item"
          >
            {{ item }}
          </span>
          <span v-if="total-page>=4">...</span>
          <span
            v-if="total-page>=4"
            :class="{'page-count-active':page===total}"
            class="page-count pointer"
            @click="page=total">{{ total }}</span>
        </div>
        <i class="el-icon-arrow-right icon-arrow pointer" @click="changePage('plus')"/>
        <div class="skip-wrapper flex flex-center">
          <span>Skip To: </span>
          <el-input
            v-model="paginationPage"
            :max="total"
            min="1"
            type="number"
            @keyup.enter.native="changePage(paginationPage)"/>
        </div>
      </div>
    </div>
    <!--<pagination/>-->
    <!--分页器-->
    <!--<pagination-->
    <!--v-show="total>0"-->
    <!--:total="total"-->
    <!--:page.sync="page"-->
    <!--:limit.sync="pageSize"-->
    <!--@pagination="getList"-->
    <!--/>-->
  </section>
</template>

<script>
import Pagination from '@/components/Pagination'

export default {
  name: 'DataOutput',
  components: {
    Pagination
  },
  props: {
    tHeader: {
      type: Array,
      default() {
        return []
      }
    },
    tBody: {
      type: Array,
      default() {
        return []
      }
    }
  },
  data() {
    return {
      page: 1,
      skip: '',
      pageSize: 10,
      paginationPage: 1
    }
  },
  computed: {
    totalArray() {
      const arr = []
      for (let i = 1; i <= this.total; i++) {
        arr.push(i)
      }
      return arr
    },
    total() {
      return Math.ceil(this.tHeader.length / this.pageSize)
    },
    header() {
      return this.sliceArray(this.tHeader)
    }
  },
  mounted() {
  },
  methods: {
    sliceArray(arr) {
      let index = 0
      const newArr = []
      while (index < arr.length) {
        newArr.push(arr.slice(index, index += this.pageSize))
      }
      return newArr
    },
    changePage(op) {
      if (op === 'plus') {
        if (this.page < this.total) {
          ++this.page
        }
      } else if (op === 'minus') {
        if (this.page > 1) {
          --this.page
        }
      } else {
        op = Number.parseInt(op)
        if (op < 1) {
          this.page = 1
          this.paginationPage = 1
        } else if (op > this.total) {
          // this.$message({
          //   type: 'warning',
          //   message: ''
          // })
          this.page = this.total
          this.paginationPage = this.total
        } else {
          this.page = op
        }
      }
    }
  }
}
</script>

<style lang="scss">
  .pagination {
    margin-top: 24px;
    font-size: 16px;
    color: #7f7d8e;
    .icon-arrow {
      width: 24px;
      height: 24px;
      margin: 0 6px;
      line-height: 24px;
      text-align: center;
    }
    .page-count {
      box-sizing: content-box;
      min-width: 14px;
      height: 24px;
      padding: 0 5px;
      margin: 0 6px;
      border-radius: 24px;
      line-height: 24px;
      text-align: center;
      &:hover {
        background: #494ece;
        color: #fff;
      }
    }
    .page-count-active {
      background: #494ece;
      color: #fff;
    }
    .skip-wrapper {
      > span {
        margin-right: 5px;
        white-space: nowrap;
      }
      .el-input__inner {
        min-width: 50px;
        height: 28px;
        padding: 0;
        padding-left: 5px;
      }
    }
  }
</style>
