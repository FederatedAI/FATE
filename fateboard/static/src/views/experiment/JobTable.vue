<template>
  <el-table
    :data="data"
    element-loading-text="Loading"
    highlight-current-row
    max-height="25vh"
  >
    <template v-for="item in tHead">
      <el-table-column
        :key="item.key"
        :prop="item.key"
        :label="item.label"
        show-overflow-tooltip
        sortable
        align="center"
      />
    </template>
  </el-table>
</template>

<script>
export default {
  name: 'JobTable',
  components: {},
  props: {
    data: {
      type: Array,
      default() {
        []
      }
    }
  },
  data() {
    return {
      tHead: [
        {
          key: 'id',
          label: 'ID'
        },
        {
          key: 'dataset',
          label: 'DATASET'
        },
        {
          key: 'type',
          label: 'TYPE'
        },
        {
          key: 'submit_time',
          label: 'SUBMIT_TIME',
          width: 150
        },
        {
          key: 'duration',
          label: 'DURATION'
        },
        {
          key: 'status',
          label: 'STATUS'
        }
      ]
    }
  }
}
</script>

<style scoped>

</style>
