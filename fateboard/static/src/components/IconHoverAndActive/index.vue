<template>
  <div
    v-show="show"
    :class="className"
    @mouseenter="mouseenter"
    @mouseout="mouseout"
    @mousedown="mousedown"
    @mouseup="mouseup"
    @click="click">
    <img :src="imgUrl" class="wh-100" alt="">
  </div>
</template>

<script>

export default {
  props: {
    show: {
      type: Boolean,
      default: true
    },
    className: {
      type: String,
      default: ''
    },
    defaultUrl: {
      type: String,
      default: ''
    },
    hoverUrl: {
      type: String,
      default: ''
    },
    activeUrl: {
      type: String,
      default: ''
    }
  },
  data() {
    return {
      status: 'default'
    }
  },
  computed: {
    imgUrl() {
      let url = ''
      if (this.status === 'default') {
        url = this.defaultUrl
      } else if (this.status === 'hover') {
        url = this.hoverUrl
      } else if (this.status === 'active') {
        url = this.activeUrl
      }
      return url
    }
  },
  methods: {
    mouseenter() {
      if (this.hoverUrl) {
        this.status = 'hover'
      }
    },
    mouseout() {
      this.status = 'default'
    },
    mousedown() {
      if (this.activeUrl) {
        this.status = 'active'
      }
    },
    mouseup() {
      if (this.status === 'active') {
        this.status = 'default'
      }
    },
    click() {
      this.$emit('clickFn')
    }
  }
}
</script>

<style scoped>

</style>
