<template>
  <div v-if="false">
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
