<template>
  <div v-loading="loading" class="running-container flex flex-center flex-col app-container">
    <ul class="job-list flex flex-center flex-wrap">
      <li v-for="item in jobList" :key="item.jobId" class="shadow">
        <div class="top flex flex-center space-between">
          <span class="job-id text-primary">{{ item.jobId }}</span>
          <span class="enter text-primary pointer" @click="enter(item.jobId)">Enter <i class="el-icon-right"/></span>
        </div>
        <div class="status pos-r flex flex-center justify-center">
          <span :style="{'font-size':item.status==='waiting' || item.status==='faied'?'14px':'36px'}" class="text pos-a text-primary">{{ item.status }}</span>
          <div
            :style="{display:item.status==='waiting'?'flex':''}"
            class="mask pos-a wh-100 flex flex-center justify-center">
            <el-button round @click="handleKillJob(item.jobId)">kill</el-button>
          </div>
          <el-progress
            :percentage="item.statusProgress"
            :show-text="false"
            :width="120"
            color="#494ece"
            type="circle"
          />
        </div>
      </li>
    </ul>
  </div>
</template>

<script>
import { getAllJobsStatus, killJob } from '@/api/job'
// import { random } from '@/utils'

export default {
  components: {},
  directives: {},
  data() {
    return {
      loading: true,
      jobList: []
    }
  },
  mounted() {
    this.getJobList()
    // console.log(process.env.WEBSOCKET_BASE_API)
    // this.initWebSocket()
  },
  methods: {
    getJobList() {
      this.jobList = []
      this.loading = true
      getAllJobsStatus().then(res => {
        this.loading = false
        res.data.forEach(job => {
          const { fJobId: jobId, fStatus: status } = job
          const progress = job.fProgress || 0
          const statusDisplay = status === 'running' ? `${progress}%` : status
          this.jobList.push({
            jobId,
            fStatus: status,
            status: statusDisplay,
            statusProgress: status === 'running' ? progress : 0
          })
        })
      })
    },
    enter(jobId) {
      this.$router.push({
        path: '/dashboard',
        query: { jobId }
      })
    },
    handleKillJob(jobId) {
      this.$confirm('You can\'t undo this action', 'Are you sure you want to kill this job?', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel',
        type: 'warning'
      }).then(() => {
        this.submitKillJob(jobId)
      }).catch(() => {
        console.log('cancel kill')
      })
    },
    submitKillJob(jobId) {
      killJob(jobId).then(res => {
        console.log('kill job:' + jobId)
        this.getJobList()
      })
    }
  }
}
</script>

<style lang="scss">
  @import "../../styles/running";
</style>
