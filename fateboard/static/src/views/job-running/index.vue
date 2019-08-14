<template>
  <div v-loading="loading" class="running-container flex flex-center flex-col app-container">
    <ul class="job-list flex flex-center flex-wrap ie-ul">
      <li v-for="item in jobList" :key="item.jobId+item.role+item.partyId" class="shadow">
        <div class="top flex flex-center space-between">
          <span class="job-id">{{ item.jobId }}</span>
          <span class="enter text-primary pointer" @click="handleKillJob(item.jobId, item.role,item.partyId)">
            <!--{{ item.status==='waiting'?'cancel':'kill' }}-->
            kill
          </span>
        </div>
        <div class="status pos-r flex flex-center justify-center">
          <span
            :style="{'font-size':item.status==='waiting' || item.status==='faied'?'14px':'36px'}"
            class="text pos-a text-primary"
          >{{ item.status }}</span>
          <div
            :style="{display:item.status==='waiting'?'flex':''}"
            class="mask pos-a wh-100 flex flex-center justify-center ie-pos"
          >
            <el-button type="text" style="font-size: 18px;" @click="enter(item.jobId,item.role,item.partyId)">
              Enter
            </el-button>
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
      getAllJobsStatus()
        .then(res => {
          res.data.forEach(job => {
            const {
              fJobId: jobId,
              fStatus: status,
              fRole: role,
              fPartyId: partyId
            } = job
            const progress = job.fProgress || 0
            const statusDisplay = status === 'running' ? `${progress}%` : status
            this.jobList.push({
              jobId,
              fStatus: status,
              status: statusDisplay,
              statusProgress: status === 'running' ? progress : 0,
              role,
              partyId
            })
          })
        })
        .then(res => {
          this.loading = false
        })
    },
    enter(job_id, role, party_id) {
      this.$router.push({
        path: '/dashboard',
        query: { job_id, role, party_id }
      })
    },
    handleKillJob(jobId, role, partyId) {
      this.$confirm('You can\'t undo this action', 'Are you sure you want to kill this job?', {
        confirmButtonText: 'Sure',
        cancelButtonText: 'Cancel'
      }).then(() => {
        this.submitKillJob({ jobId, role, partyId })
      }).catch(() => {
        // console.log('cancel kill')
      })
    },
    submitKillJob({ jobId, role, partyId }) {
      const para = { job_id: jobId, role, party_id: partyId }
      killJob(para).then(res => {
        // console.log('kill job:' + jobId)
        this.getJobList()
      })
    }
  }
}
</script>

<style lang="scss">
  @import '../../styles/running';

  .ie-ul {
    width: 100%;
  }

  .ie-pos {
    top: 0;
    left: 0;
  }
</style>
