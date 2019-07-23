<template>
  <el-scrollbar wrap-class="scrollbar-wrapper">
    <el-menu
      :default-active="$route.path"
      :collapse="isCollapse"
      :background-color="variables.menuBg"
      :text-color="variables.menuText"
      :active-text-color="variables.menuActiveText"
      :collapse-transition="false"
      mode="vertical"
    >
      <el-menu-item>
        <div class="my-first-item" @click="gotoHome">
          <img v-if="!isCollapse" src="@/assets/webank.svg" height="13px">
          <span v-if="!isCollapse">客户价值管理平台</span>
          <img v-if="isCollapse" src="@/assets/webank-logo.png" width="25px">
        </div>
      </el-menu-item>
      <sidebar-item v-for="route in routes" :key="route.path" :item="route" :base-path="route.path"/>
    </el-menu>
  </el-scrollbar>
</template>

<script>
import { mapGetters } from 'vuex'
import variables from '@/styles/variables.scss'
import SidebarItem from './SidebarItem'

export default {
  components: { SidebarItem },
  computed: {
    ...mapGetters([
      'sidebar'
    ]),
    routes() {
      return this.$router.options.routes.filter(item => item.path !== '/')
    },
    variables() {
      return variables
    },
    isCollapse() {
      return !this.sidebar.opened
    }
  },
  methods: {
    gotoHome() {
      this.$router.push({ path: '/old-job-dashboard' })
    }
  }
}
</script>
<style lang="scss" scoped>
.my-first-item{
  width:100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  span{
    margin-top:5px;
    line-height:16px;
    color: white;
    font-size:14px;
  }
}
</style>
