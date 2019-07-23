<template>
  <div class="app-container exp-container">
    <!--头部：面包屑和search按钮-->
    <div class="app-top flex space-between flex-center">
      <el-breadcrumb separator-class="el-icon-arrow-right text-black">
        <el-breadcrumb-item>Datasets overview</el-breadcrumb-item>
      </el-breadcrumb>
      <el-button @click="dialogVisible=true">CREATE</el-button>
    </div>
    <!--表格-->
    <div class="table-wrapper">
      <el-table
        v-loading="listLoading"
        :data="list"
        element-loading-text="Loading"
        highlight-current-row
        max-height="75vh"
      >

        <el-table-column type="expand">
          <template slot-scope="props">
            <job-table :data="props.row.jobs"/>
          </template>
        </el-table-column>

        <template v-for="item in tHead">
          <el-table-column
            v-if="item.key==='jobs'"
            :key="item.key"
            :prop="item.key"
            :label="item.label"
            show-overflow-tooltip
            sortable
            align="center"
          >
            <template slot-scope="scope">
              {{ scope.row.jobs.length }}
            </template>
          </el-table-column>

          <el-table-column
            v-else-if="!item.hidden"
            :key="item.key"
            :prop="item.key"
            :label="item.label"
            show-overflow-tooltip
            sortable
            align="center"/>
        </template>
        <el-table-column label="OPTIONS" align="center" width="300">
          <template slot-scope="scope">
            <el-button size="small" @click="editExp(scope.row)">EDIT</el-button>
            <el-button type="danger" size="small" @click="deleteExp(scope.row)">DELETE</el-button>
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

    <!--新增界面 start-->
    <el-dialog
      :visible.sync="dialogVisible"
      :close-on-click-modal="false"
      title="Add dataset"
      width="35%">
      <el-form ref="form" :model="form" :rules="formRules" label-width="120px" label-position="left">
        <el-form-item label="name" prop="name" class="el-form-item-required">
          <el-input :maxlength="20" v-model="form.name"/>
        </el-form-item>
        <el-form-item label="description" prop="desc" class="el-form-item-required">
          <el-input :maxlength="50" v-model="form.desc"/>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button :loading="formLoading" @click="handleAdd">Create</el-button>
        <el-button @click.native="dialogVisible = false">cancel</el-button>
      </div>
    </el-dialog>
    <!--新增界面 end-->
  </div>
</template>

<script>
import Pagination from '@/components/Pagination'
import JobTable from './JobTable'
import { parseTime } from '@/utils'

export default {
  name: 'Experiment',
  components: {
    Pagination,
    JobTable
  },
  data() {
    return {
      // 表格相关数据和属性
      list: null,
      tHead: [
        {
          key: 'id',
          label: 'ID'
        },
        {
          key: 'name',
          label: 'NAME'
        },
        {
          key: 'partner',
          label: 'PARTNER'
        },
        {
          key: 'create_time',
          label: 'CREATE_TIME',
          width: 150
        },
        {
          key: 'jobs',
          label: 'JOBS'
        }
      ],
      listLoading: false,
      pageSize: 10,
      total: 0,
      page: 1,
      // create对话框相关属性
      dialogVisible: false,
      formLoading: false,
      form: {
        name: '',
        type: '',
        desc: ''
      },
      // 新增表单规则（必选限制）
      formRules: {
        name: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        desc: [
          { required: true, message: 'Please enter a description', trigger: 'blur' }
        ]
      }
    }
  },
  mounted() {
    this.getList()
  },
  methods: {
    // 查询表格数据
    getList() {
      const pno = this.page
      // getChannelList(this.filters).then(res => {
      //   const fn = () => {
      //     this.listLoading = false
      let data = []
      for (let i = 0; i < 50; i++) {
        const index = i + 1
        data.push({
          id: index,
          name: `testexp${index}`,
          partner: `partner${index}`,
          create_time: parseTime(new Date()),
          jobs: [
            {
              id: 'jobid',
              dataset: 'lib1.table1',
              type: 'intersection',
              submit_time: parseTime(new Date()),
              duration: parseTime(new Date(), '{h}:{i}:{s}'),
              status: 'complete'
            },
            {
              id: 'jobid',
              dataset: 'lib1.table1',
              type: 'model training',
              submit_time: parseTime(new Date()),
              duration: parseTime(new Date(), '{h}:{i}:{s}'),
              status: 'failed'
            },
            {
              id: 'jobid',
              dataset: 'lib1.table1',
              type: 'intersection',
              submit_time: parseTime(new Date()),
              duration: parseTime(new Date(), '{h}:{i}:{s}'),
              status: 'complete'
            }
          ]
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
    },
    // 删除实验
    deleteExp(row) {
      this.$message({ message: '删除成功' })
    },
    // 编辑实验
    editExp(row) {
      this.$router.push('/editExperiment')
    },
    handleAdd() {
      console.log('添加实验成功')
      this.$router.push('/createExperiment')
    }
  }
}
</script>

<style lang="scss" scoped>
  @import "src/styles/exp";
</style>
