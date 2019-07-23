<template>
  <div v-loading="projectListLoading" class="dashboard-container bg-dark">
    <!--顶部添加按钮-->
    <div class="dashboard-top flex flex-end flex-center">
      <el-button @click="openAddDialog">create a project</el-button>
    </div>
    <!--project列表-->
    <ul class="project-list flex flex-wrap">
      <li v-loading="item.loading" v-for="(item,index) in projectsData" :key="index">
        <project-item
          :data="item"
          @openEditDialog="openEditDialog"
          @deleteProject="deleteProject(item.pid,index)"
        />
      </li>
    </ul>
    <!--新增/修改界面 start-->
    <el-dialog
      :visible.sync="dialogVisible"
      :close-on-click-modal="false"
      :title="dialogTitle"
      width="35%">
      <el-form ref="form" :model="form" :rules="formRules" label-width="120px" label-position="left">
        <el-form-item label="name" prop="name" class="el-form-item-required">
          <el-input :maxlength="40" v-model="form.name"/>
        </el-form-item>
        <el-form-item label="type" prop="type" class="el-form-item-required">
          <el-select v-model="form.type" placeholder="Please select type">
            <el-option
              v-for="item in $store.getters.projectType"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>

        </el-form-item>
        <el-form-item label="description" prop="desc" class="el-form-item-required">
          <el-input :maxlength="200" v-model="form.desc" type="textarea"/>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button :loading="dialogLoading" @click="handleCommitForm">{{ dialogBtnText }}</el-button>
        <el-button @click.native="dialogVisible = false">cancel</el-button>
      </div>
    </el-dialog>
    <!--新增/修改界面 end-->
  </div>
</template>

<script>
// import { mapGetters } from 'vuex'
import ProjectItem from './ProjectItem'
import { getProjectList, addProject, updateProject, deleteProject } from '@/api/project'

export default {
  name: 'Dashboard',
  components: {
    ProjectItem
  },
  data() {
    return {
      projectsData: [],
      projectListLoading: false,
      dialogVisible: false,
      dialogLoading: false,
      form: {
        name: '',
        type: '',
        desc: ''
      },
      formStatus: 'add',
      // 新增表单规则（必选限制）
      formRules: {
        name: [
          { required: true, message: 'Please enter your name', trigger: 'blur' }
        ],
        type: [
          { required: true, message: 'Please enter a type', trigger: 'blur' }
        ],
        desc: [
          { required: true, message: 'Please enter a description', trigger: 'blur' }
        ]
      }
    }
  },
  computed: {
    dialogTitle() {
      return this.formStatus === 'add' ? 'new project' : 'edit project'
    },
    dialogBtnText() {
      return this.formStatus === 'add' ? 'Create' : 'Save'
    }
  },
  mounted() {
    this.getList()
  },
  methods: {
    // 获取project列表数据
    getList() {
      this.projectListLoading = true
      getProjectList().then(res => {
        this.projectListLoading = false
        this.projectsData = res.data
        this.projectsData.map(item => {
          item.loading = false
        })
      }).catch(err => {
        this.projectListLoading = true
        console.log(err)
      })
    },
    // 点击新增project或save（对话框中）
    handleCommitForm() {
      this.$refs.form.validate((valid) => {
        if (valid) {
          this.dialogVisible = false
          if (this.formStatus === 'add') {
            addProject(this.form).then(res => {
              this.$message({ message: '新增成功', type: 'success' })
              // this.getList()
              this.$router.push({
                path: '/data-center',
                query: {
                  pid: res.data.pid, pname: this.form.name
                }
              })
            })
          } else {
            updateProject(this.form).then(res => {
              this.$message({ message: '编辑', type: 'success' })
              this.getList()
            })
          }
        }
      })
    },
    // 初始化新增表单
    initForm() {
      this.form = {
        name: '',
        type: '',
        desc: ''
      }
    },
    // 打开新建Project对话框
    openAddDialog() {
      this.formStatus = 'add'
      this.initForm()
      this.dialogVisible = true
    },
    // 打开编辑Project对话框
    openEditDialog(pid) {
      this.formStatus = 'edit'
      this.projectsData.forEach(project => {
        if (project.pid === pid) {
          this.form = {
            pid,
            name: project.name,
            type: project.type,
            desc: project.desc
          }
        }
      })
      this.dialogVisible = true
    },
    deleteProject(pid, index) {
      this.projectsData[index].loading = true
      // 数据无法从数组中属性更新至视图，需要强制用set方法或splice
      // this.$set(this.projectsData, index, this.projectsData[index])
      this.projectsData.splice(index, 1, this.projectsData[index])
      deleteProject({ pid }).then(res => {
        this.projectsData[index].loading = false
        this.$set(this.projectsData, index, this.projectsData[index])
        this.$message({ message: '删除成功', type: 'success' })
        this.getList()
      })
    }
  }
}
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
  .dashboard-container {
    height: 100%;
    padding: 0 calc((100vw - (75vw + (15px * 10))) / 2);
    .dashboard-top {
      height: 100px;
      padding-right: 15px;
    }
    .project-list {
      margin: auto;
      max-height: 75vh;
      overflow: auto;
      > li {
        margin: 0 15px 25px;
      }
    }
  }
</style>
