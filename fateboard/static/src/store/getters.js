const getters = {
  sidebar: state => state.app.sidebar,
  device: state => state.app.device,
  isOpenReqSimulate: state => state.app.isOpenReqSimulate,
  projectType: state => state.app.projectType,
  jobType: state => state.app.jobType

  // token: state => state.user.token,
  // avatar: state => state.user.avatar,
  // name: state => state.user.name,
  // roles: state => state.user.roles
}
export default getters
