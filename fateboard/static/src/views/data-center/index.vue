<template>
  <el-row class="data-center-container">

    <!--侧边栏 start-->
    <el-col :span="4" class="side-bar h-100">
      <!--添加dataset按钮-->
      <div class="add-dataset">
        <el-button @click="addDialogVisible = true">ADD DATASET</el-button>
      </div>
      <!--树形节点-->
      <div class="menu-tree">
        <el-tree
          :data="sideBarData"
          :props="sideBarProps"
          :default-expand-all="true"
          :expand-on-click-node="false"
          @node-click="handleNodeClick"/>
      </div>
    </el-col>
    <!--侧边栏 end-->

    <!--主界面 start-->
    <el-col :span="20" class="main-view h-100 bg-dark">
      <!--头部：面包屑和search按钮-->
      <div class="app-top">
        <el-breadcrumb separator-class="el-icon-arrow-right text-black">
          <el-breadcrumb-item>Datasets overview</el-breadcrumb-item>
          <el-breadcrumb-item>lib1</el-breadcrumb-item>
        </el-breadcrumb>
        <el-button>search</el-button>
      </div>
      <!--表格-->
      <div class="table-wrapper">
        <el-table
          v-loading="listLoading"
          :data="list"
          element-loading-text="Loading"
          fit
          highlight-current-row
          max-height="75vh"
        >
          <template v-for="item in tHead">
            <el-table-column
              v-if="!item.hidden"
              :key="item.key"
              :prop="item.key"
              :label="item.label"
              :formatter="item.formatter"
              show-overflow-tooltip
              sortable
              align="center"/>
          </template>
          <el-table-column label="操作" align="center" width="300">
            <template slot-scope="scope">
              <el-button size="small" @click="previewLib(scope.row)">PREVIEW</el-button>
              <el-button type="danger" size="small" @click="deleteLib(scope.row)">DELETE</el-button>
            </template>
          </el-table-column>
        </el-table>
        <!--分页器-->
        <pagination
          v-show="total>0"
          :total="total"
          :page.sync="page"
          :limit.sync="pageSize"
          @pagination="getList"
        />
      </div>
    </el-col>
    <!--主界面 end-->

    <!--选择dataset对话框-->
    <choose-dataset
      :visible="addDialogVisible"
      :title="'Add dataset'"
      @closeDialog="addDialogVisible = false"/>
    <!--preview 对话框-->
    <PreviewLib :visible="previewDialogVisible" @closeDialog="previewDialogVisible = false"/>
  </el-row>
</template>

<script>
import Pagination from '@/components/Pagination'
import PreviewLib from '@/views/components/PreviewLib'
import ChooseDataset from '@/views/components/ChooseDataset'
import { parseTime } from '@/utils'

export default {
  name: 'DataCenter',
  components: {
    Pagination,
    PreviewLib,
    ChooseDataset
  },
  data() {
    return {
      // 侧边栏数据，属性
      sideBarData: [

        {
          // label: {
          //   name:'lib1',
          //   type:2, // 1:all 2:lib 3:table
          //   isCheked:true,
          //   id:'libid1'
          // },
          label: 'lib1',
          children: [
            {
              label: 'table1'
            },
            {
              label: 'table2'
            },
            {
              label: 'table4'
            }
          ]
        },
        {
          label: 'lib2',
          children: [
            {
              label: 'table2'
            },
            {
              label: 'table3'
            }
          ]
        },
        {
          label: 'lib3',
          children: [
            {
              label: 'table2'
            }
          ]
        }
      ],
      sideBarProps: {
        children: 'children',
        label: 'label'
      },
      // 表格相关数据和属性
      list: null,
      tHead: [
        {
          key: 'id',
          hidden: true
        },
        {
          key: 'index',
          label: 'INDEX'
        },
        {
          key: 'lib',
          label: 'LIBRARY'
        },
        {
          key: 'table',
          label: 'TABLE'
        },
        {
          key: 'size',
          label: 'SIZE'
        },
        {
          key: 'rows',
          label: 'ROWS'
        },
        {
          key: 'cols',
          label: 'COLUMNS'
        },
        {
          key: 'upload_time',
          label: 'UPLOAD_TIME',
          width: 150
        }
      ],
      listLoading: false,
      pageSize: 10,
      total: 0,
      page: 1,
      // add dataset对话框相关属性
      // preview 对话框显示
      addDialogVisible: false,
      addLoading: false,
      previewDialogVisible: false
    }
  },
  mounted() {
    this.getList()
  },
  methods: {
    // 点击节点
    handleNodeClick(data) {
      console.log('点击节点:', data.label)
    },
    // 展示Lib
    previewLib(row) {
      console.log('preview:', row)
      this.previewDialogVisible = true
    },
    // 删除Lib
    deleteLib(row) {
      this.$message({ message: '删除成功', type: 'success' })
    },
    // 查询表格数据
    getList() {
      const pno = this.page
      // getChannelList(this.filters).then(res => {
      //   const fn = () => {
      //     this.listLoading = false
      let data = []
      for (let i = 0; i < 50; i++) {
        data.push({
          index: i + 1,
          id: i,
          lib: 'lib1',
          table: 'table' + i,
          size: '6666KB',
          rows: 666,
          cols: 777,
          upload_time: parseTime(new Date(), '{y}.{m}.{d}')
        })
      }

      if (Array.isArray(data)) {
        // 分页
        this.total = data.length
        data = data.filter((row, index) => {
          return index < this.pageSize * pno && index >= this.pageSize * (pno - 1)
        })
        this.list = data
        //     }
        //   }
        //   // setTimeout(fn, 500)
        //   fn()
        // }).catch(err => {
        //   console.log('发生错误：', err)
        // })
      }
    }
  }
}
</script>

<style lang="scss" scoped>
  @import "src/styles/dataCenter";
</style>
