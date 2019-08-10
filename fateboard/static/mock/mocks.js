import user from './user'
import channels from './channelsManage'
import projects from './project'
import datasets from './dataset'
import experiments from './experiment'
import jobs from './job'

export default [
  ...user,
  ...channels,
  ...projects,
  ...datasets,
  ...experiments,
  ...jobs
]

