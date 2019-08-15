const getters = {
  sidebar: state => state.app.sidebar,
  modelNameMap: state => state.app.modelNameMap,
  metricTypeMap: state => state.app.metricTypeMap,
  device: state => state.app.device,
  isOpenReqSimulate: state => state.app.isOpenReqSimulate,
  jobType: state => state.app.jobType,
  icons: state => state.app.icons,
  currentCvTab: state => state.app.currentCvTab,
  evaluationFlags: state => state.app.evaluationFlags,
  evaluationInstances: state => state.app.evaluationInstances
  // token: state => state.user.token,
  // avatar: state => state.user.avatar,
  // name: state => state.user.name,
  // roles: state => state.user.roles
}
export default getters
