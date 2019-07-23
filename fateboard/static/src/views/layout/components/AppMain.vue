<template>
  <section class="app-main bg-dark">
    <transition name="fade-transform" mode="out-in">
      <!-- or name="fade" -->
      <!-- <router-view :key="key"></router-view> -->
      <router-view/>
    </transition>
  </section>
</template>

<script>
export default {
  name: 'AppMain',
  computed: {
    // key() {
    //   return this.$route.name !== undefined ? this.$route.name + +new Date() : this.$route + +new Date()
    // }
  }
}
</script>

<style scoped>
.app-main {
  display: flex;
  flex-direction: column;
  align-items: center;
  /*48 = navbar高度  */
  height: calc(100vh - 48px);
  position: relative;
  overflow: auto;
  padding: 0 50px;
}
</style>
