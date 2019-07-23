<template>
  <el-dialog
    :visible.sync="visible"
    :close-on-click-modal="false"
    :before-close="close"
    title="Add partner"
    width="35%"
  >
    <el-form ref="form" :model="form" :rules="formRules" label-width="120px" label-position="left">
      <el-form-item
        v-for="(item,index) in formOptions"
        :key="index"
        :label="item.label"
        :prop="item.prop"
        class="el-form-item-required"
      >
        <el-select v-model="form[item.prop]" style="width: 90%;">
          <el-option
            v-for="option in item.options"
            :key="option.value"
            :value="option.value"
            :label="option.label"
          />
        </el-select>
      </el-form-item>
    </el-form>
    <div slot="footer" class="dialog-footer">
      <el-button :loading="loading" @click="handleCommitForm">Save</el-button>
      <el-button @click.native="close">cancel</el-button>
    </div>
  </el-dialog>
</template>

<script>
export default {
  name: 'AddPartner',
  components: {},
  props: {
    visible: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      loading: false,
      form: {
        partner: '',
        lib: '',
        table: ''
      },
      formOptions: [
        {
          prop: 'partner',
          label: 'Partner',
          options: [{ value: '', label: 'please select' }, { value: 1, label: 'tencent' }, { value: 2, label: 'webank' }]
        },
        {
          prop: 'lib',
          label: 'Library',
          options: [{ value: '', label: 'please select' }, { value: 1, label: 'lib1' }, {
            value: 2,
            label: 'lib2'
          }, { value: 3, label: 'lib3' }]
        },
        {
          prop: 'table',
          label: 'Table',
          options: [
            { value: '', label: 'please select' },
            { value: 1, label: 'table1' },
            { value: 2, label: 'table2' },
            { value: 3, label: 'table3' },
            { value: 4, label: 'table4' }
          ]
        }
      ],
      // 新增表单规则（必选限制）
      formRules: {
        partner: [
          { required: true, message: 'Please choose your partner', trigger: 'change' }
        ],
        lib: [
          { required: true, message: 'Please choose a library', trigger: 'change' }
        ],
        table: [
          { required: true, message: 'Please choose a table', trigger: 'change' }
        ]
      }
    }
  },
  mounted() {

  },
  methods: {
    handleCommitForm() {
      this.$message({ message: '新增成功', type: 'success' })
      this.close()
    },
    close() {
      this.$emit('closeDialog', this.visible)
      return false
    },
    initForm() {
      this.form = {
        partner: 0,
        lib: 0,
        desc: 0
      }
    }
  }
}
</script>

<style scoped>

</style>
