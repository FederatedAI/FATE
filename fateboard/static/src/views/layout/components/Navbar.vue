<template>
  <div class="nav-container shadow bg-dark-primary flex">
    <!--导航栏主内容-->
    <div class="navbar bg-primary flex space-between flex-center">
      <div>
        <div class="home-btn c-fff" @click="go('/')"><b>FATE</b>Board</div>
      </div>
      <div class="router-btns flex c-fff">
        <span :class="{'active':path === '/running'}" @click="go('/running')">RUNNING</span>
        <span :class="{'active':path === '/history'}" @click="go('/history')">JOBS</span>
      </div>
      <!--<el-switch-->
      <!--v-model="isSimulate"-->
      <!--style="display: block;margin-right: 20px;"-->
      <!--active-color="#409eff"-->
      <!--inactive-color="#ff4949"-->
      <!--active-text="打开模拟弱网络请求(1秒延时)"-->
      <!--inactive-text="关闭模拟"-->
      <!--@change="changeSimulate"-->
      <!--/>-->
    </div>
    <!--帮助（问号图标）-->
    <div class="help flex flex-center justify-center">
      <span class="icon">?</span>
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
// import { getProjectList } from '@/api/project'

export default {
  components: {},
  data() {
    return {
      projectsData: null,
      isSimulate: true,
      path: this.$route.path
    }
  },
  computed: {
    routes() {
      // console.log(this.$route.fullPath === '/old-job-dashboard')
      return this.$router.options.routes[0].children.filter(item => {
        return item.path === '/data-center' || item.path === '/experiment' || item.path === '/job-history'
      })
    },
    ...mapGetters([
      'isOpenReqSimulate'
    ]),
    pname() {
      return this.$route.query.pname || ''
    },
    pid() {
      return this.$route.query.pid || ''
    }
  },
  watch: {
    '$route': 'getPath'
  },
  mounted() {
    this.isSimulate = this.isOpenReqSimulate
  },
  methods: {
    getPath() {
      this.path = this.$route.path
    },
    go(path) {
      this.$router.push(path)
      this.path = path
    },
    // 选择project回调
    handleCommand(command) {
      // 刷新当前路由
      // this.$router.replace({
      //   path: '/refresh',
      //   query: { pid: command.pid }
      // })
      this.$router.push({
        path: this.$route.path,
        query: { pid: command.pid, pname: command.pname }
      })
    },
    // 切换模拟
    changeSimulate(flag) {
      this.$store.dispatch('SwitchReqSimulate', flag)
    }
  }
}
</script>

<style lang="scss" scoped>
  @import "../../../styles/navbar";
</style>

