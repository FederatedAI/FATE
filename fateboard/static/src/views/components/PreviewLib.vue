<template>
  <el-dialog
    :visible.sync="visible"
    :close-on-click-modal="false"
    :before-close="close"
    title="Dataset preview"
    width="80%"
  >
    <div>
      <el-table
        v-loading="loading"
        :data="list"
        fit
        element-loading-text="Loading"
        highlight-current-row
        max-height="50vh"
      >
        <el-table-column
          v-for="(item,index) in tHead"
          :key="index"
          :label="item.name"
          align="center"
        >
          <el-table-column
            :label="item.type"
            :prop="item.name"
            width="120"
            show-overflow-tooltip
            align="center"
          />
        </el-table-column>
      </el-table>

      <!--分页器-->
      <pagination
        v-show="total>0"
        :total="total"
        :page.sync="page"
        :limit.sync="pageSize"
        @pagination="getList"
      />
    </div>
  </el-dialog>
</template>

<script>
import Pagination from '@/components/Pagination'

export default {
  name: 'Preview',
  components: {
    Pagination
  },
  props: {
    visible: {
      type: Boolean,
      default: false
    }
  },
  data() {
    return {
      loading: false,
      tHead: [
        {
          name: 'name',
          type: 'type'
        },
        {
          name: 'age',
          type: 'int'
        },
        {
          name: 'skin',
          type: 'string'
        },
        {
          name: 'bmi',
          type: 'double'
        },
        {
          name: 'ear',
          type: 'double'
        },
        {
          name: 'leg',
          type: 'string'
        },
        {
          name: 'hand',
          type: 'string'
        },
        {
          name: 'hair',
          type: 'string'
        },
        {
          name: 'snake',
          type: 'string'
        },
        {
          name: 'tiger',
          type: 'string'
        },
        {
          name: 'elephant',
          type: 'string'
        },
        {
          name: 'cat',
          type: 'string'
        },
        {
          name: 'dog',
          type: 'string'
        },
        {
          name: 'apple',
          type: 'string'
        },
        {
          name: 'boy',
          type: 'string'
        },
        {
          name: 'girl',
          type: 'string'
        },
        {
          name: 'pear',
          type: 'string'
        },
        {
          name: 'man',
          type: 'string'
        },
        {
          name: 'woman',
          type: 'string'
        },
        {
          name: 'mate',
          type: 'string'
        },
        {
          name: 'lien',
          type: 'string'
        },
        {
          name: 'fish',
          type: 'string'
        },
        {
          name: 'computer',
          type: 'string'
        },
        {
          name: 'pen',
          type: 'string'
        },
        {
          name: 'phone',
          type: 'string'
        },
        {
          name: 'pad',
          type: 'string'
        },
        {
          name: 'cup',
          type: 'string'
        },
        {
          name: 'cap',
          type: 'string'
        },
        {
          name: 'shy',
          type: 'string'
        }
      ],
      pageSize: 10,
      total: 0,
      page: 1,
      list: null
    }
  },
  mounted() {
    this.getList()
  },
  methods: {
    // 查询表格数据
    getList() {
      const pno = this.page
      // getChannelList(this.filters).then(res => {
      //   const fn = () => {
      //     this.listLoading = false
      let data = []
      for (let i = 0; i < 50; i++) {
        data.push({
          name: i + 1,
          id: i,
          age: `age${i}`,
          skin: `skin${i}`,
          bmi: `bmi${i}`,
          apple: `apple${i}`,
          ear: `ear${i}`,
          leg: `leg${i}`,
          hand: `hand${i}`,
          pear: `pear${i}`,
          dog: `dog${i}`,
          elephant: `elephant${i}`,
          tiger: `tiger${i}`,
          snake: `snake${i}`,
          boy: `boy${i}`,
          girl: `girl${i}`,
          man: `man${i}`,
          lion: `lion${i}`
        })
      }

      if (Array.isArray(data)) {
        // 分页
        this.total = data.length
        data = data.filter((row, index) => {
          return index < this.pageSize * pno && index >= this.pageSize * (pno - 1)
        })
        this.list = data
        //     }
        //   }
        //   // setTimeout(fn, 500)
        //   fn()
        // }).catch(err => {
        //   console.log('发生错误：', err)
        // })
      }
    },
    close() {
      this.$emit('closeDialog', this.visible)
      return false
    }
  }
}
</script>

<style scoped>

</style>
