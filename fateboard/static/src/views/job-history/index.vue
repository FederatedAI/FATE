<template>
  <div class="app-container history-container bg-dark">
    <!--头部-->
    <h3 class="app-title">Job Overview</h3>
    <!--表格-->
    <div v-loading="listLoading" class="table-wrapper">
      <el-table
        :data="list"
        :row-class-name="tableRowClassName"
        :header-row-class-name="'t-header'"
        element-loading-text="Loading"
        highlight-current-row
        height="70vh"
      >
        <template v-for="item in tHead">
          <el-table-column
            v-if="!item.hidden"
            :key="item.key"
            :prop="item.key"
            :label="item.label"
            :sortable="item.sortable"
            show-overflow-tooltip
            border
          >
            <template slot-scope="scope">
              <span v-if="item.key==='jobId'" class="text-primary pointer" @click="toDetailes(scope.row[item.key])">{{ scope.row[item.key] }}</span>
              <el-progress
                v-else-if="item.key==='status' && scope.row.progress"
                :percentage="scope.row.progress"
                color="#494ece"/>
              <span v-else>{{ scope.row[item.key] }}</span>
            </template>
          </el-table-column>
        </template>

      </el-table>
      <!--分页器-->
      <pagination
        :total="total"
        :page.sync="page"
        :layout="'prev, pager, next'"
        :limit.sync="pageSize"
        @pagination="handlePageChange"
      />

    </div>

  </div>
</template>

<script>
import Pagination from '@/components/Pagination'
import { parseTime } from '@/utils'

import { getAllJobs, getJobsTotal } from '@/api/job'

export default {
  name: 'Job',
  components: {
    Pagination
  },
  filters: {
    formatType(type) {
      let typePresent = '未知'
      switch (type) {
        case 1:
          typePresent = 'intersection'
          break
        case 2:
          typePresent = 'feature engineering'
          break
        case 3:
          typePresent = 'model training'
          break
        case 4:
          typePresent = 'model prdiction'
          break
      }
      return typePresent
    }
  },
  data() {
    return {
      // 表格相关数据和属性
      list: null,
      tHead: [
        {
          key: 'jobId',
          label: 'jobId'
        },
        {
          key: 'dataset',
          label: 'DATASET'
        },
        {
          key: 'partner',
          label: 'PARTNER'
        },
        {
          key: 'pnr_dataset',
          label: 'PNR-DATASET',
          width: 180
        },
        {
          key: 'start_time',
          label: 'START TIME',
          width: 180
        },
        {
          key: 'end_time',
          label: 'END TIME',
          width: 180
        },
        {
          key: 'duration',
          label: 'DURATION',
          width: 150
        },
        {
          key: 'status',
          label: 'STATUS',
          width: 150
        },
        {
          key: 'progress',
          hidden: true,
          width: 150
        }
      ],
      listLoading: true,
      pageSize: 10,
      total: 0,
      page: 1,
      // create对话框相关属性
      dialogVisible: false,
      formLoading: false,
      form: {
        experiment: '',
        type: '',
        desc: ''
      },
      // 新增表单规则（必选限制）
      formRules: {
        experiment: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        type: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        desc: [
          { required: true, message: 'Please enter a description', trigger: 'blur' }
        ]
      }
    }
  },
  mounted() {
    this.getTotal()
    // this.getList()
  },
  methods: {
    getTotal() {
      getJobsTotal().then(res => {
        this.total = res.data
        if (!this.list) {
          this.getList()
        }
      })
    },
    handlePageChange({ page }) {
      this.page = page
      this.getList()
    },

    getList() {
      this.listLoading = true
      const para = {
        total: this.total,
        pno: this.page,
        psize: this.pageSize
      }
      getAllJobs(para).then(res => {
        this.listLoading = false
        const data = []
        res.data.list.forEach(item => {
          let jobId = ''
          let _dataset = ''
          let partner = ''
          let pnr_dataset = ''
          let start_time = ''
          let end_time = ''
          let duration = ''
          let status = ''
          let progress = ''

          const { job, dataset } = item

          if (job) {
            jobId = job.fJobId || ''
            start_time = job.fStartTime ? parseTime(new Date(job.fStartTime)) : ''
            end_time = job.fEndTime ? parseTime(job.fEndTime) : ''
            duration = job.fStartTime ? parseTime(job.fStartTime, '{h}:{i}:{s}') : ''
            status = job.fStatus || ''
            progress = job.fStatus || job.fStatus === 'running' ? job.fProgress : null
          }
          if (dataset) {
            _dataset = dataset.dataset || ''
            partner = dataset.partner || ''
            pnr_dataset = dataset.pnr_dataset || ''
          }
          data.push({
            jobId,
            dataset: _dataset,
            partner,
            pnr_dataset,
            start_time,
            end_time,
            duration,
            status,
            progress
          })
        })
        this.list = data
        // if (Array.isArray(data)) {
        //   this.total = data.length
        //   data = data.filter((row, index) => {
        //     return index < this.pageSize * pno && index >= this.pageSize * (pno - 1)
        //   })
        //   // console.log(data)
        //   this.list = data
        // }
      })
    },

    deleteExp(row) {
      this.$message({ message: '删除成功' })
    },
    toDetailes(jobId) {
      this.$router.push({
        path: '/details',
        query: { jobId, 'from': 'Job overview' }
      })
    },
    tableRowClassName({ row, rowIndex }) {
      if (rowIndex % 2 === 0) {
        // console.log(rowIndex)
        return 't-row history-stripe'
      }
      return 't-row'
    }
  }
}
</script>

<style lang="scss">
  .history-container {
    /*padding-top: 40px;*/
    .table-wrapper {
      /*height: 70vh;*/
      box-shadow: 0 3px 10px 1px #ddd;
    }
    .t-header {
      height: 64px;
      color: #494ece;
      font-size: 16px;
      font-weight: bold;
      text-align: left;
    }
    .t-row {
      height: 56px;
      font-size: 16px;
      color: #7f7f8e;
    }
    .el-table {
      padding: 0 30px;
    }
    .el-table .history-stripe {
      background: #f8f8fa;
    }
  }

</style>
