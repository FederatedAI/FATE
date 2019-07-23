<template>
  <el-dialog
    :visible.sync="visible"
    :close-on-click-modal="false"
    :title="title"
    :before-close="close"
    width="35%"
  >
    <div class="tree-view">
      <el-tree
        ref="tree"
        :data="data"
        :default-expand-all="true"
        :expand-on-click-node="false"
        :default-checked-keys="[1]"
        node-key="id"
        show-checkbox
      />
    </div>
    <div slot="footer" class="dialog-footer">
      <el-button :loading="loading" @click="addSubmit">ok</el-button>
      <el-button @click.native="close">cancel</el-button>
    </div>
  </el-dialog>
</template>

<script>
export default {
  name: 'ChooseDataset',
  props: {
    visible: {
      type: Boolean,
      default: false
    },
    title: {
      type: String,
      default: 'Choose dataset'
    }
  },
  data() {
    return {
      loading: false,

      data: [
        {
          label: 'all',
          children: [
            {
              label: 'lib1',
              children: [
                {
                  id: 1,
                  label: 'table1'
                },
                {
                  id: 2,
                  label: 'table2'
                },
                {
                  label: 'table3'
                },
                {
                  label: 'table4'
                }
              ]
            },
            {
              label: 'lib2',
              children: [
                {
                  id: 3,
                  label: 'table1'
                },
                {
                  id: 4,
                  label: 'table2'
                },
                {
                  label: 'table3'
                },
                {
                  label: 'table4'
                },
                {
                  label: 'table5'
                },
                {
                  label: 'table6'
                }
              ]
            },
            {
              label: 'lib3',
              children: [
                {
                  label: 'table1'
                },
                {
                  label: 'table2'
                }
              ]
            }
          ]
        }
      ],
      props: {
        label: 'label',
        children: 'children'
      }
    }
  },
  methods: {
    close() {
      this.$emit('closeDialog', this.visible)
      return false
    },
    // 监听切换添加dataset树形结构
    handleCheckChange(a, b, c) {
      console.log('check change', a, b, c)
    },
    // 添加dataset
    addSubmit() {
      console.log(this.$refs.tree.getCheckedNodes())
      console.log(this.$refs.tree.getCheckedKeys())
      this.$message({ message: '新增成功', type: 'success' })
      this.close()
    }
  }
}
</script>

<style scoped>
  .tree-view {
    overflow-y: auto;
    height: 50vh;
  }
</style>
