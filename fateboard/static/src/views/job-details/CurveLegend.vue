<template>
  <div class="legend-wrapper flex flex-center">
    <icon-hover-and-active
      :class-name="'page-arrow'"
      :default-url="icons.normal.left"
      :hover-url="icons.hover.left"
      :active-url="icons.active.left"
      :show="legendData.length>pageSize"
      @clickFn="prePage"
    />
    <ul class="legend-list flex flex-center">
      <li v-for="(item,index) in fiterLegendData" :key="index">
        <div v-if="Array.isArray(item)" class="flex flex-col">
          <div
            class="legend-group flex flex-center"
            style="margin-bottom: 12px;"
            @click="clickLegend(item[0].text,index)">
            <span :style="{'background-color':item.isActive?item[0] && item[0].color:'#bbbbc8'}" class="color"/>
            <p :style="{'color':item.isActive?'':'#bbbbc8'}" class="text">{{ item[0].text }}</p>
          </div>
          <div
            class="legend-group flex flex-center"
            @click="clickLegend(item[1].text,index)">
            <span :style="{'background-color':item.isActive?item[1] && item[1].color:'#bbbbc8'}" class="color"/>
            <p :style="{'color':item.isActive?'':'#bbbbc8'}" class="text">{{ item[1].text }}</p>
          </div>
        </div>
        <div v-else>
          <div class="legend-group flex flex-center" @click="clickLegend(item.text,index)">
            <span :style="{'background-color':item.isActive?item.color:'#bbbbc8'}" class="color"/>
            <p :style="{'color':item.isActive?'':'#bbbbc8'}" class="text">{{ item.text }}</p>
          </div>
        </div>
      </li>
    </ul>
    <icon-hover-and-active
      :class-name="'page-arrow'"
      :default-url="icons.normal.right"
      :hover-url="icons.hover.right"
      :active-url="icons.active.right"
      :show="legendData.length>pageSize"
      @clickFn="nextPage"
    />
  </div>
</template>

<script>
import IconHoverAndActive from '@/components/IconHoverAndActive'
import { mapGetters } from 'vuex'

export default {
  name: 'DataOutput',
  components: {
    IconHoverAndActive
  },
  props: {
    legendData: {
      type: Array,
      default() {
        return []
      }
    },
    instanceIndex: {
      type: Number,
      default: -1
    }
  },
  data() {
    return {
      page: 1,
      pageSize: 3
    }
  },
  computed: {
    ...mapGetters([
      'icons',
      'evaluationInstances'
    ]),
    fiterLegendData() {
      return this.legendData.slice((this.page - 1) * this.pageSize, this.page * this.pageSize)
    }
  },
  mounted() {
    for (let i = 0; i < this.legendData.length; i++) {
      this.legendData[i].isActive = true
    }
    this.legendData.splice()
  },
  methods: {
    prePage() {
      if (this.page > 1) {
        --this.page
      }
    },
    nextPage() {
      if (this.page < Math.ceil(this.legendData.length / this.pageSize)) {
        ++this.page
      }
    },
    clickLegend(curveName, legendIndex) {
      // curveName.replace(/(_tpr|_fpr)/g, '')
      legendIndex = (this.page - 1) * this.pageSize + legendIndex
      this.legendData[legendIndex].isActive = !this.legendData[legendIndex].isActive
      this.legendData.splice()
      this.$emit('clickLegend', { curveName, evaluationListIndex: this.instanceIndex })
    }
  }
}
</script>

<style lang="scss" scoped>
  .legend-wrapper {
    .page-arrow {
      width: 24px;
      height: 24px;
      cursor: pointer;
      &:first-of-type {
        margin-right: 24px;
      }
      &:last-of-type {
        margin-left: 24px;
      }
    }
    .legend-list {
      > li {
        margin-right: 32px;
        &:last-of-type {
          margin-right: 0;
        }
        .legend-group {
          cursor: pointer;
          .color {
            $w: 18px;
            width: $w;
            height: $w;
            margin-right: 5px;
            border-radius: 1px;
          }
          .text {
            color: #7f7d8e;
          }
        }
      }
    }
  }
</style>
