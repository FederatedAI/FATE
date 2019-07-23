<template>
  <section class="project-container flex flex-col space-around pos-r">
    <!--设置按钮，点击出现下拉框，选择编辑或删除-->
    <!--<i class="iconfont iconshezhi pos-a"/>-->
    <el-dropdown
      class="pos-a"
      trigger="click"
      @command="handleCommand">
      <span class="el-dropdown-link">
        <i class="iconfont iconshezhi"/>
      </span>
      <!--project下拉选-->
      <el-dropdown-menu slot="dropdown">
        <el-dropdown-item command="edit">edit</el-dropdown-item>
        <el-dropdown-item command="delete">delete</el-dropdown-item>
      </el-dropdown-menu>
    </el-dropdown>
    <div class="project-msg">
      <h2 class="name">{{ data.name }}</h2>
      <p class="time">time: {{ data.time }}</p>
      <p class="type">type: {{ data.type | projectTypeFormat }}</p>
      <!--<p class="type">type: {{ $store.getters.projectType[data.type].label }}</p>-->
      <p class="desc">{{ data.desc }}</p>
    </div>

    <ul class="data-bar flex space-between text-primary t-a-c">
      <li class="data-view pointer" @click="go('/data-center')">
        <p class="name">datasets</p>
        <p class="count">{{ data.datasets }}</p>
      </li>
      <li class="data-view pointer" @click="go('/experiment')">
        <p class="name">experiments</p>
        <p class="count">{{ data.experiments }}</p>
      </li>
      <li class="data-view pointer" @click="go('/job-history')">
        <p class="name">jobs</p>
        <p class="count">{{ data.jobs }}</p>
      </li>
    </ul>
  </section>
</template>

<script>
export default {
  name: 'ProjectItem',
  props: {
    data: {
      type: Object,
      default() {
        return null
      }
    }
  },
  data() {
    return {}
  },
  methods: {
    go(path) {
      this.$router.push({
        path,
        query: {
          pid: this.data.pid,
          pname: this.data.name
        }
      })
    },
    // 选择project回调
    handleCommand(command) {
      switch (command) {
        case 'edit':
          this.$emit('openEditDialog', this.data.pid)
          break
        case 'delete':
          this.$confirm('you can\'t undo this action', 'Would you like to delete this project?', {
            confirmButtonText: 'Yes',
            cancelButtonText: 'Cancel'
          }).then(() => {
            this.$emit('deleteProject')
          })
          break
      }
    }
  }
}
</script>

<style lang="scss">
  .project-container {
    width: 25vw;
    /*height: 15vw;*/
    min-height: 265px;
    padding: 30px 10px;
    background: #fff;
    .project-msg {
      padding: 0 30px;
      .name {
        margin-bottom: 20px;
      }
      .time, .type, .desc {
        line-height: 22px;
      }
      .desc {
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
        text-overflow: ellipsis;
        overflow: hidden;
        overflow-wrap: break-word;
      }
    }

    .el-dropdown {
      right: 20px;
      top: 20px;
      cursor: pointer;
    }
    .data-bar {
      .data-view {
        width: 33%;
        .name {

        }
        .count {
          font-size: 25px;

        }
      }
    }
  }
</style>
