<template>
  <div class="nav-container shadow bg-dark-primary flex">
    <div class="navbar bg-primary flex space-between flex-center">
      <div>
        <div class="home-btn c-fff" @click="go('/')"><b>FATE</b>Board</div>
      </div>
      <div class="router-btns flex">
        <span :class="{'active':path === '/running'}" @click="go('/running')">RUNNING</span>
        <span :class="{'active':path === '/history'}" @click="go('/history')">JOBS</span>
      </div>
    </div>

    <div class="help flex flex-center justify-center">
      <!--<span class="icon" @click="openModal"><i class="el-icon-s-tools"/></span>-->
      <span @click="openModal"><svg
        t="1565061740972"
        class="icon"
        viewBox="0 0 1025 1024"
        version="1.1"
        xmlns="http://www.w3.org/2000/svg"
        p-id="4168"
        width="200"
        height="200"><path
          d="M1021.452956 443.829303c-2.94308-38.196064-10.044861-41.650984-47.793065-38.643924-41.778944 3.32696-77.095908-17.466542-91.491409-53.871165-14.779381-37.236364-4.09472-75.624367 28.471103-102.30403 21.945142-17.978382 24.632302-32.565823 4.73452-53.615245-29.942643-31.670103-61.164886-62.252546-93.730709-91.171509-25.272102-22.393002-37.044424-19.513902-58.413746 7.421681-26.103843 32.885723-67.882787 42.674664-105.82293 24.952202-35.572883-16.634802-52.271665-47.025305-49.520525-90.147829 1.85542-29.558763-6.845861-39.923524-36.340644-42.162824-23.032802-1.79144-46.065604-3.00706-67.690847-4.41462C481.141803 2.36726 459.964421 4.22268 438.978979 7.037801 409.612156 10.940581 403.278136 19.513902 405.837336 49.584505 409.740116 95.202249 390.738054 129.495533 352.861891 145.170634 316.137367 160.397876 278.133243 149.457295 249.278261 115.419931c-17.082662-20.153702-31.990003-23.096782-50.864105-5.246361C166.680072 139.988254 135.521809 170.698657 106.602847 203.32846 81.906564 231.159763 84.657704 240.628803 113.192787 264.429366 140.32031 287.078288 152.092631 316.061231 144.54299 351.122274 136.80141 386.759138 114.216467 409.66398 78.899504 417.917401 65.911563 420.924461 51.644021 419.005061 38.01628 418.877101 19.462078 418.685161 7.753737 427.130522 5.322497 445.300843-1.395404 495.717088-1.651324 546.197313 7.497817 596.421618c3.58288 19.833802 14.267541 27.511403 34.293283 26.295783C52.347801 622.077601 63.032462 619.582381 73.397223 620.606061c36.980444 3.64686 66.027366 28.215183 76.008247 62.828366 9.980881 34.421243-1.85542 69.546267-31.542143 93.410809-24.440362 19.641862-27.127523 33.269603-5.886161 55.854545 29.430803 31.286223 59.949266 61.740706 92.195189 90.083849 27.447423 24.120462 37.876164 21.305342 60.781006-7.933521 31.414183-40.179444 87.716589-45.553764 126.232552-12.604061 25.847923 22.137082 32.757763 50.096345 29.366823 82.598188-2.49522 23.992502 6.334021 37.812184 27.383443 38.260044 47.281225 0.9597 94.690409 1.15164 141.843674-1.66348 30.710403-1.85542 36.788504-12.796001 33.333583-43.762324-5.05442-44.913964 13.819681-80.166948 51.503905-96.161949 38.068104-16.122962 76.648047-5.182381 104.86323 29.750703 17.274602 21.369322 32.437863 24.504342 52.015745 5.822181 32.437863-30.902343 63.980006-62.956326 93.730709-96.417869 21.625242-24.312402 18.554202-37.428304-7.485661-57.070166-21.753202-16.378882-34.037363-38.132084-35.572883-65.195626-3.199-55.342705 41.906904-97.31359 97.95339-91.747329 28.599063 2.81512 42.610684-5.310341 43.250484-30.262543C1024.460016 532.249672 1024.843896 487.847548 1021.452956 443.829303zM513.387726 735.002312c-122.969572-1.34358-221.626742-101.40831-220.027241-223.162262 1.5995-120.154452 100.76851-218.939581 219.259481-218.427741 122.457732 0.51184 223.482162 101.21637 222.202562 221.690722C733.606908 637.176882 633.670138 736.345892 513.387726 735.002312z"
          p-id="4169"
          fill="#ffffff"/></svg></span>
    </div>
    <SSHConfig
      :show-config-modal="showConfigModal"
      @closeSSHConfigModal="showConfigModal = false"
    />
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import SSHConfig from './SSHConfig'
// import { getAllSSHConfig, getSSHConfig, removeSSHConfig, addSSHConfig } from '@/api/ssh'

export default {
  components: {
    SSHConfig
  },
  data() {
    return {
      projectsData: null,
      isSimulate: true,
      path: this.$route.path,
      showConfigModal: false
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
    handleCommand(command) {
      // this.$router.replace({
      //   path: '/refresh',
      //   query: { pid: command.pid }
      // })
      this.$router.push({
        path: this.$route.path,
        query: { pid: command.pid, pname: command.pname }
      })
    },
    changeSimulate(flag) {
      this.$store.dispatch('SwitchReqSimulate', flag)
    },
    openModal() {
      this.showConfigModal = true
    },
    save() {
      this.$message('save successfully')
      this.showConfigModal = false
    }
  }
}
</script>

<style lang="scss" scoped>
  @import "../../../styles/navbar";
</style>

