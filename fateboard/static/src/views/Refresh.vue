<template>
  <div v-if="false">这是一个空的页面，作为刷新路由重定向使用
  </div>
</template>

<script>
export default {
  name: 'Refresh',
  beforeRouteEnter(to, from, next) {
    next(vm => {
      console.log(from)
      vm.$router.replace(from.path)
    })
  }
}
</script>

<style scoped>

</style>
