<template>
  <div class="app-container channels-container">
    <!--顶部工具条-->
    <el-col class="toolbar" style="padding-bottom: 0px;">
      <el-form :inline="true" :model="filters" style="text-align: left;">
        <el-form-item label="渠道名称">
          <el-input
            v-model="filters.channel"
            placeholder="请输入渠道名称"
            @keyup.native.prevent="keyupQuery"
          />
        </el-form-item>
        <el-form-item label="渠道值">
          <el-input
            v-model="filters.channelId"
            placeholder="请输入渠道值"
            @keyup.native.prevent="keyupQuery"
          />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="fetchData">查询</el-button>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleAdd">新增</el-button>
        </el-form-item>
        <!--<el-form-item>-->
        <!--<el-button type="primary" @click="addRequest">测试按钮</el-button>-->
        <!--</el-form-item>-->
      </el-form>
    </el-col>
    <!--列表-->
    <el-table
      v-loading="listLoading"
      :data="list"
      element-loading-text="Loading"
      border
      fit
      highlight-current-row>
      <template v-for="item in tHead">
        <el-table-column
          v-if="!item.hidden"
          :key="item.key"
          :prop="item.key"
          :label="item.label"
          show-overflow-tooltip
          align="center"
          width="200"/>
      </template>
      <el-table-column label="操作" align="center" width="200">
        <template slot-scope="scope">
          <el-button type="primary" size="small" @click="handleUpdate(scope.row)">修改</el-button>
          <el-button type="danger" size="small" @click="handleDelete(scope.row)">删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!--分页器-->
    <!--<el-pagination-->
    <!--:total="total"-->
    <!--:page-size="pageSize"-->
    <!--layout="prev, pager, next"-->
    <!--style="padding: 20px;"-->
    <!--@current-change="handleCurrentChange"-->
    <!--/>-->

    <pagination
      v-show="total>0"
      :total="total"
      :page.sync="page"
      :limit.sync="pageSize"
      @pagination="fetchData"
    />

    <!--新增界面-->
    <el-dialog
      :visible.sync="addFormVisible"
      :close-on-click-modal="false"
      title="新增渠道"
      width="35%">
      <el-form ref="addForm" :model="addForm" :rules="addFormRules" label-width="120px" label-position="left">
        <el-form-item label="渠道名称" prop="channel" class="el-form-item-required">
          <el-input :maxlength="50" v-model="addForm.channel"/>
        </el-form-item>
        <el-form-item label="渠道值" prop="channelId" class="el-form-item-required">
          <el-input :maxlength="20" v-model="addForm.channelId"/>
        </el-form-item>
        <el-form-item label="子渠道名称" prop="subChannel" class="el-form-item-required">
          <el-input :maxlength="50" v-model="addForm.subChannel"/>
        </el-form-item>
        <el-form-item label="子渠道值" prop="subChannelId" class="el-form-item-required">
          <el-input :maxlength="20" v-model="addForm.subChannelId"/>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <!--<el-button @click.native="addFormVisible = false">取消</el-button>-->
        <el-button :loading="addLoading" type="primary" @click="addSubmit">确定
        </el-button>
      </div>
    </el-dialog>

    <!--修改界面-->
    <el-dialog :visible.sync="updateFormVisible" :close-on-click-modal="false" title="修改渠道" width="35%">
      <el-form ref="updateForm" :model="updateForm" :rules="updateFormRules" label-width="120px" label-position="left">
        <el-form-item label="渠道名称" prop="channel" class="el-form-item-required">
          <el-input v-model="updateForm.channel"/>
        </el-form-item>
        <!--<el-form-item label="渠道值" prop="channelId" class="el-form-item-required">-->
        <!--<el-input v-model="updateForm.channelId" :disabled="true"/>-->
        <!--</el-form-item>-->
        <el-form-item label="子渠道名称" prop="subChannel" class="el-form-item-required">
          <el-input v-model="updateForm.subChannel"/>
        </el-form-item>
        <!--<el-form-item label="子渠道值" prop="subChannelId" class="el-form-item-required">-->
        <!--<el-input v-model="updateForm.subChannelId" :disabled="true"/>-->
        <!--</el-form-item>-->
      </el-form>
      <div slot="footer" class="dialog-footer">
        <!--<el-button @click.native="addFormVisible = false">取消</el-button>-->
        <el-button :loading="updateLoading" type="primary" @click="updateSubmit">确定</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import { getChannelList, addChannel, deleteChannel, updateChannel } from '@/api/channelsManage'
import Pagination from '@/components/Pagination' // Secondary package based on el-pagination
// import { parseTime } from '@/utils'

export default {
  components: { Pagination },
  data() {
    return {
      filters: {
        channel: '',
        channelId: ''
      },
      list: null,
      tHead: [
        {
          key: 'code',
          hidden: true
        },
        {
          key: 'channel',
          label: '渠道名称'
        },
        {
          key: 'channelId',
          label: '渠道值'
        },
        {
          key: 'subChannel',
          label: '子渠道名称'
        },
        {
          key: 'subChannelId',
          label: '子渠道值'
        },
        {
          key: 'createTime',
          label: '操作时间'
        }],
      listLoading: true,
      pageSize: 10,
      total: 0,
      page: 1,
      // 新增界面是否显示
      addFormVisible: false,
      addLoading: false,
      addFormRules: {
        channel: [
          { required: true, message: '请输入渠道名称', trigger: 'blur' }
        ],
        channelId: [
          { required: true, message: '请输入渠道值', trigger: 'blur' }
        ],
        subChannel: [
          { required: true, message: '请输入子渠道名称', trigger: 'blur' }
        ],
        subChannelId: [
          { required: true, message: '请输入子渠道值', trigger: 'blur' }
        ]
      },
      // 新增界面数据
      addForm: {
        channel: '',
        channelId: '',
        subChannel: '',
        subChannelId: ''
      },
      // 编辑界面是否显示
      updateFormVisible: false,
      updateLoading: false,
      updateFormRules: {
        channel: [
          { required: true, message: '请输入渠道名称', trigger: 'blur' }
        ],
        channelId: [
          { required: true, message: '请输入渠道值', trigger: 'blur' }
        ],
        subChannel: [
          { required: true, message: '请输入子渠道名称', trigger: 'blur' }
        ],
        subChannelId: [
          { required: true, message: '请输入子渠道值', trigger: 'blur' }
        ]
      },
      // 编辑界面数据
      updateForm: {
        code: '',
        channel: '',
        channelId: '',
        subChannel: '',
        subChannelId: ''
      }
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    /**
     * 查询列表数据
     */
    fetchData() {
      const pno = this.page
      this.listLoading = true
      getChannelList(this.filters).then(res => {
        const fn = () => {
          this.listLoading = false
          let data = res.data
          if (Array.isArray(data)) {
            // 分页
            this.total = data.length
            data = data.filter((row, index) => {
              return index < this.pageSize * pno && index >= this.pageSize * (pno - 1)
            })
            this.list = data
          }
        }
        // Just to simulate the time of the request
        // setTimeout(fn, 500)
        fn()
      }).catch(err => {
        console.log('发生错误：', err)
      })
    },

    /**
     * 新建按钮
     */
    handleAdd() {
      this.addFormVisible = true
      this.addForm = {
        channel: '',
        channelId: '',
        subChannel: '',
        subChannelId: ''
      }
    },

    /**
     * 删除按钮
     * @param row 行数据
     */
    handleDelete(row) {
      this.$confirm('确认删除该渠道吗', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消'
        // 点击确定，删除渠道
      }).then(() => {
        this.deleteRequest(row.code)
      })
    },

    /**
     * 修改按钮
     */
    handleUpdate(row) {
      this.updateFormVisible = true
      this.updateForm = {
        code: row.code,
        channel: row.channel,
        channelId: row.channelId,
        subChannel: row.subChannel,
        subChannelId: row.subChannelId
      }
    },

    /**
     * 新增form确定按钮
     */
    addSubmit(code) {
      this.$refs.addForm.validate((valid) => {
        if (valid) {
          this.$confirm('确认新增吗', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消'
            // 点击确定，新增渠道
          }).then(() => {
            this.addRequest(code)
          })
        }
      })
    },

    /**
     * 修改form确定按钮
     */
    updateSubmit() {
      this.$refs.updateForm.validate((valid) => {
        if (valid) {
          this.$confirm('确认修改吗', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消'
            // 点击确定，新增渠道
          }).then(() => {
            this.updateRequest()
          })
        }
      })
    },

    /**
     * 新增请求
     */
    addRequest() {
      addChannel(this.addForm).then(res => {
        this.addLoading = true
        const fn = () => {
          this.addLoading = false
          this.$message({
            message: '新增成功',
            type: 'success'
          })
          // 关闭对话框，并清空新增表单
          this.addFormVisible = false
          this.addForm = {
            channel: '',
            channelId: '',
            subChannel: '',
            subChannelId: ''
          }
          this.fetchData()
        }

        // Just to simulate the time of the request
        // setTimeout(fn, 1000)
        fn()
      }).catch(err => {
        this.addLoading = false
        console.log('出错啦！错误：', err)
      })
    },

    /**
     * 删除请求
     */
    deleteRequest(code) {
      deleteChannel({ code }).then(res => {
        this.listLoading = true
        const fn = () => {
          this.listLoading = false
          this.$message({
            message: '删除成功',
            type: 'success'
          })
          this.fetchData()
        }

        // Just to simulate the time of the request
        // setTimeout(fn, 1000)
        fn()
      }).catch(err => {
        this.listLoading = false
        console.log('出错啦！错误：', err)
      })
    },

    /**
     * 修改请求
     */
    updateRequest() {
      this.updateLoading = true
      updateChannel(this.updateForm).then(res => {
        const fn = () => {
          this.updateLoading = false
          this.$message({
            message: '修改成功',
            type: 'success'
          })
          // 关闭对话框，并清空新增表单
          this.updateFormVisible = false
          this.updateForm = {
            code: '',
            channel: '',
            channelId: '',
            subChannel: '',
            subChannelId: ''
          }
          this.fetchData()
        }

        // Just to simulate the time of the request
        // setTimeout(fn, 1000)
        fn()
      }).catch(err => {
        this.updateLoading = false
        console.log('出错啦！错误：', err)
      })
    },

    /**
     * 处理当前页数变化
     * @param pno 当前页数
     */
    handleCurrentChange(pno) {
      this.page = pno
      this.fetchData()
    },

    /**
     * 键盘回车查询
     * @param e 按下键盘事件
     */
    keyupQuery(e) {
      const { keyCode } = e
      if (keyCode === 13) {
        this.fetchData()
      }
    }
  }
}
</script>

<style lang="scss">
  .channels-container {
    .el-dialog__title {
      font-weight: bold;
    }

    .el-form-item__label {
      font-weight: normal;
    }

    .el-form {
      .el-form-item {
        margin-bottom: 5px;
        .el-input__inner {
          height: 35px;
        }
      }
      .el-form-item-required {
        margin-bottom: 22px;
      }
      .el-form-item__label {
        font-weight: normal;
      }
    }
  }
</style>
