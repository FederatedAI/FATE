<template>
  <div class="app-container history-container bg-dark">
    <h3 class="app-title">Job Overview</h3>

    <div v-loading="listLoading" class="table-wrapper">
      <el-table
        :data="list"
        :row-class-name="tableRowClassName"
        :header-row-class-name="'t-header'"
        fit
        element-loading-text="Loading"
        highlight-current-row
        empty-text="NO DATA"
        height="70vh"
      >
        <template v-for="item in tHead">
          <el-table-column
            v-if="!item.hidden"
            :key="item.key"
            :prop="item.key"
            :label="item.label"
            :width="item.width"
            :min-width="item.minWidth"
            :sortable="item.sortable"
            show-overflow-tooltip
            border
          >
            <template slot-scope="scope">
              <span
                v-if="item.key==='jobId'"
                class="text-primary pointer"
                @click="toDetailes(scope.row[item.key],scope.row['role'],scope.row['partyId'])"
              >{{ scope.row[item.key] }}</span>
              <div v-else-if="item.key==='status'">
                <div v-if="scope.row.progress || scope.row.progress===0">
                  <!--<el-progress-->
                  <!--:percentage="scope.row.progress"-->
                  <!--:show-text="true"-->
                  <!--color="#494ece"-->
                  <!--/>-->
                  <div class="progress-wrapper flex flex-center">
                    <div class="progress-bg">
                      <div :style="{width:`${scope.row.progress}%`}" class="progress-block" />
                    </div>
                    <span class="progress-text">{{ scope.row.progress }}%</span>
                  </div>
                </div>
                <div v-else>{{ scope.row[item.key] }}</div>
                <!--<span class="text-primary" style="font-size: 12px;">{{ scope.row.progress }}%</span>-->
              </div>
              <span v-else>{{ scope.row[item.key] }}</span>
            </template>
          </el-table-column>
        </template>
      </el-table>
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
import { parseTime, formatSeconds } from '@/utils'

import { getAllJobs, getJobsTotal } from '@/api/job'

export default {
  name: 'Job',
  components: {
    Pagination
  },
  filters: {
    formatType(type) {
      let typePresent = ''
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
      list: null,
      tHead: [
        {
          key: 'jobId',
          label: 'ID',
          minWidth: 300
        },
        {
          key: 'role',
          label: 'Role',
          width: 100
        },
        {
          key: 'partyId',
          label: 'Party ID',
          width: 100
        },
        {
          key: 'start_time',
          label: 'Start Time',
          width: 180
        },
        {
          key: 'end_time',
          label: 'End Time',
          width: 180
        },
        {
          key: 'duration',
          label: 'Duration',
          width: 150
        },
        {
          key: 'status',
          label: 'Status',
          width: 220
        },
        {
          key: 'progress',
          hidden: true
          // width: 150
        }
      ],
      listLoading: true,
      pageSize: 20,
      total: 0,
      page:
				(this.$route.params.page && Number.parseInt(this.$route.params.page)) ||
				1,
      dialogVisible: false,
      formLoading: false,
      form: {
        experiment: '',
        type: '',
        desc: ''
      },
      formRules: {
        experiment: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        type: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        desc: [
          {
            required: true,
            message: 'Please enter a description',
            trigger: 'blur'
          }
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
      const para = {
        total: this.total,
        pno: this.page,
        psize: this.pageSize
      }
      getAllJobs(para)
        .then(res => {
          const data = []
          res.data.list.forEach(item => {
            let jobId = ''
            let role = ''
            let partyId = ''
            // let _dataset = ''
            // let partner = ''
            // let pnr_dataset = ''
            let start_time = ''
            let end_time = ''
            let duration = ''
            let status = ''
            let progress = ''

            const { job } = item

            if (job) {
              jobId = job.fJobId || ''
              role = job.fRole || ''
              partyId = job.fPartyId || ''
              start_time = job.fStartTime
                ? parseTime(new Date(job.fStartTime))
                : ''
              end_time = job.fEndTime ? parseTime(job.fEndTime) : ''
              duration = job.fElapsed ? formatSeconds(job.fElapsed) : ''
              status = job.fStatus || ''
              progress = job.fStatus === 'running' ? job.fProgress || 0 : null
            }
            // if (dataset) {
            //   _dataset = dataset.dataset || ''
            //   partner = dataset.partner || ''
            //   pnr_dataset = dataset.pnr_dataset || ''
            // }
            data.push({
              jobId,
              role,
              partyId,
              // dataset: _dataset,
              // partner,
              // pnr_dataset,
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
        .then(res => {
          this.listLoading = false
        })
    },

    deleteExp(row) {
      this.$message({ message: 'delete success' })
    },
    toDetailes(job_id, role, party_id) {
      this.$router.push({
        path: '/details',
        query: { job_id, role, party_id, from: 'Job overview', page: this.page }
      })
    },
    tableRowClassName({ row, rowIndex }) {
      // if (rowIndex % 2 === 0) {
      //   // console.log(rowIndex)
      //   return 't-row history-stripe'
      // }
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
		color: #534c77;
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
	.progress-wrapper {
		.progress-bg {
			$h: 5px;
			width: 50%;
			height: $h;
			border-radius: $h;
			background: #e8e8ef;
			overflow: hidden;
			.progress-block {
				height: 100%;
				background: #494ece;
			}
		}
		.progress-text {
			margin-left: 7px;
			color: #494ece;
			font-size: 16px;
		}
	}
}
</style>
