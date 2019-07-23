<template>
  <div class="app-container create-exp-container">
    <!--头部：面包屑和search按钮-->
    <div class="app-top flex space-between flex-center">
      <el-breadcrumb separator-class="el-icon-arrow-right text-black">
        <el-breadcrumb-item>Create experiment</el-breadcrumb-item>
      </el-breadcrumb>
    </div>
    <hr>
    <!--实验详情-->
    <section class="item-section">
      <p class="section-title">About experiment</p>
      <div class="section-view">
        <ul class="msg-list flex flex-wrap">
          <li>Experiment ID: {{ expData.id }}</li>
          <li>Experiment name: {{ expData.name }}</li>
          <li>Submit time: {{ expData.submitTime }}</li>
          <li>Description: {{ expData.desc }}</li>
        </ul>
      </div>
    </section>
    <!--上传lib-->
    <section class="item-section">
      <p class="section-title">Dataset {{ isCreateRoute ? 'Upload' : 'Overview' }}</p>
      <div class="section-view flex justify-center">
        <el-button v-if="isCreateRoute" @click="addDialogVisible = true">Choose dataset</el-button>
        <div v-else class="overview flex space-between flex-center w-100">
          <h3>{{ overview.lib }} . {{ overview.table }}</h3>
          <ul class="overview-list flex space-between flex-center">
            <li v-for="(item,index) in overview.data" :key="index" class="list">
              <p class="key">{{ item.key }}</p>
              <p class="value">{{ item.value }}</p>
            </li>
          </ul>
          <div class="flex flex-center">
            <el-button @click="previewDialogVisible = true">PREVIEW</el-button>
            <el-button @click="deleteTable">DELETE</el-button>
          </div>
        </div>
      </div>
    </section>
    <!--选择Partner-->
    <section class="item-section">
      <p class="section-title">Datasets from partners</p>
      <div class="section-view flex justify-center">
        <el-button v-if="isCreateRoute" @click="partnerDialogVisible = true">Add partner</el-button>
        <div v-else class="overview flex space-between flex-center w-100">
          <h3>{{ partner.lib }} . {{ partner.table }}</h3>
          <div class="overview-list flex space-between flex-center">
            <div class="list">
              <p class="key">from</p>
              <p class="value">{{ partner['from'] }}</p>
            </div>
          </div>
          <el-button @click="deletePartner">DELETE</el-button>
        </div>
      </div>
    </section>
    <!--上传json文件-->
    <section class="item-section">
      <p class="section-title">Experiment setting</p>
      <div class="section-view flex justify-center">
        <el-upload
          ref="upload"
          :before-upload="beforeUpload"
          drag
          accept=".json"
          action="">
          <p><span class="text-primary">ADD FILE</span> (OR DRAG & DROP)</p>
          <!--<el-button-->
          <!--:disabled="fileList.length<=0"-->
          <!--style="margin-left: 10px;"-->
          <!--size="small"-->
          <!--type="success"-->
          <!--@click="submitUpload">上传到服务器-->
          <!--</el-button>-->
          <div v-if="currentUploadFileName" slot="tip" class="t-a-c">upload success: <span class="text-primary">{{ currentUploadFileName }}</span>
          </div>
          <div v-else slot="tip" class="el-upload__tip t-a-c">File formats supported: JSON</div>
        </el-upload>
      </div>
    </section>
    <!--底部按钮-->
    <div class="tool-bar flex justify-center">
      <el-button class="btn" @click="save">Save</el-button>
      <el-button class="btn" @click="startJob">Start Job Manager</el-button>
    </div>
    <!--选择dataset对话框-->
    <choose-dataset
      :visible="addDialogVisible"
      :title="'Add dataset'"
      @closeDialog="addDialogVisible = false"
    />
    <!--选择partner对话框-->
    <add-partner :visible="partnerDialogVisible" @closeDialog="partnerDialogVisible = false"/>
    <!--preview 对话框-->
    <PreviewLib :visible="previewDialogVisible" @closeDialog="previewDialogVisible = false"/>
  </div>
</template>

<script>
import ChooseDataset from '@/views/components/ChooseDataset'
import PreviewLib from '@/views/components/PreviewLib'
import AddPartner from './AddPartner'
import { parseTime } from '../../utils'

export default {
  name: 'Experiment',
  components: {
    ChooseDataset,
    AddPartner,
    PreviewLib
  },
  data() {
    return {
      // 实验详细信息初始化
      expData: {},
      // 选择dataset对话框属性
      addDialogVisible: false,
      addLoading: false,
      previewDialogVisible: false,
      currentUploadFileName: '',
      partnerDialogVisible: false,
      overview: {
        lib: 'lib1',
        table: 'table1',
        data: [
          {
            key: 'size',
            value: '355KB'
          },
          {
            key: 'rows',
            value: 6666
          },
          {
            key: 'columns',
            value: 777
          }
        ]
      },
      partner: {
        lib: 'lib2',
        table: 'table2',
        'from': 'Tencent'
      }
    }
  },
  computed: {
    isCreateRoute() {
      return this.$route.path === '/createExperiment'
    }
  },
  mounted() {
    this.getExpData()
  },
  methods: {
    getExpData() {
      this.expData = {
        id: 'E0001',
        name: 'testExp',
        submitTime: parseTime(new Date()),
        desc: 'an experiment that...'
      }
    },
    // 上传方法
    beforeUpload(file) {
      console.log('beforeUpload:', file)
      if (file.size > 1024 * 1024 * 1) {
        this.$message({
          type: 'warning',
          message: '请上传小于1MB的文件'
        })
        return
      }
      this.currentUploadFileName = file.name
      this.$message({ message: '上传成功', type: 'success' })
      const formData = new FormData()
      formData.append('file', file)
      console.log(formData)
      // this.dialogLoading = true
      // uploadImg(formData).then(res => {
      //   this.dialogLoading = false
      //   this.dialogForm.content = res.data
      //   this.$message({
      //     message: '上传成功',
      //     type: 'success'
      //   })
      // }).catch(err => {
      //   this.dialogLoading = false
      //   console.log('上传失败', err)
      // })

      return false
    },
    // 保存实验
    save() {
      this.$message({ message: '保存实验成功', type: 'success' })
    },
    // 保存并启动Job
    startJob() {
      this.save()
      this.$router.push('/jobSetting')
    },
    // 删除表
    deleteTable() {
      this.$message({ message: '删除table成功', type: 'success' })
    },
    // 删除partner
    deletePartner() {
      this.$message({ message: '删除partner成功', type: 'success' })
    }
  }
}
</script>

<style lang="scss">
  @import "src/styles/exp";
</style>
