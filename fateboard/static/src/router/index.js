import Vue from 'vue'
import Router from 'vue-router'

// in development-env not use lazy-loading, because lazy-loading too many pages will cause webpack hot update too slow. so only in production use lazy-loading;
// detail: https://panjiachen.github.io/vue-element-admin-site/#/lazy-loading

Vue.use(Router)

/* Layout */
import Layout from '../views/layout/Layout'

/**
 * hidden: true                   if `hidden:true` will not show in the sidebar(default is false)
 * alwaysShow: true               if set true, will always show the root menu, whatever its child routes length
 *                                if not set alwaysShow, only more than one route under the children
 *                                it will becomes nested mode, otherwise not show the root menu
 * redirect: noredirect           if `redirect:noredirect` will no redirect in the breadcrumb
 * name:'router-name'             the name is used by <keep-alive> (must set!!!)
 * meta : {
    title: 'title'               the name show in subMenu and breadcrumb (recommend set)
    icon: 'svg-name'             the icon show in the sidebar
    breadcrumb: false            if false, the item will hidden in breadcrumb(default is true)
  }
 **/
export const constantRouterMap = [
  {
    path: '/',
    component: Layout,
    redirect: '/running',
    name: 'Dashboard',
    hidden: true,
    children: [
      {
        path: '/running',
        name: 'RUNNINNG',
        component: () => import('@/views/job-running')
      },
      {
        path: '/dashboard',
        component: () => import('@/views/job-dashboard/index')
      },
      // {
      //   path: '/refresh',
      //   component: () => import('@/views/Refresh')
      // },
      // {
      //   path: '/data-center',
      //   name: 'DATA CENTER',
      //   component: () => import('@/views/data-center')
      // },
      // {
      //   path: '/experiment',
      //   name: 'EXPERIMENT',
      //   component: () => import('@/views/experiment')
      // },
      {
        path: '/history',
        name: 'HISTORY',
        component: () => import('@/views/job-history')
      },
      // {
      //   path: '/createExperiment',
      //   name: 'CreateExperiment',
      //   component: () => import('@/views/create-experiment')
      // },
      // {
      //   path: '/editExperiment',
      //   name: 'editExperiment',
      //   component: () => import('@/views/create-experiment')
      // },
      // {
      //   path: '/jobSetting',
      //   name: 'JobSetting',
      //   component: () => import('@/views/job-history-setting')
      // },
      {
        path: '/details',
        name: 'JobDetails',
        component: () => import('@/views/job-details')
      }
      // {
      //   path: '/oldDashboard',
      //   name: 'Dashboard',
      //   component: () => import('@/views/old-job-dashboard')
      // }
    ]
  },
  // { path: '/login', component: () => import('@/views/login/index'), hidden: true },
  { path: '/404', component: () => import('@/views/404'), hidden: true },
  { path: '*', redirect: '/404', hidden: true }
]

const router = new Router({
  // mode: 'history', //后端支持可开
  scrollBehavior: () => ({ y: 0 }),
  routes: constantRouterMap
})

export default router
