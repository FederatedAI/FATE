import Cookies from 'js-cookie'

const app = {
  state: {
    sidebar: {
      opened: !+Cookies.get('sidebarStatus'),
      withoutAnimation: false
    },
    device: 'desktop',
    isOpenReqSimulate: Cookies.get('isOpenReqSimulate') !== 'false',
    modelNameMap: {
      homoLR: 'HomoLR',
      heteroLR: 'HeteroLR',
      dataIO: 'DataIO',
      evaluation: 'Evaluation',
      boost: 'HeteroSecureBoost',
      binning: 'HeteroFeatureBinning',
      selection: 'HeteroFeatureSelection',
      scale: 'FeatureScale',
      sample: 'FederatedSmple',
      oneHot: 'OneHotEncoder'
    },
    metricTypeMap: {
      dataIOTable: 'DATAIO_TABLE',
      scale: 'SCALE',
      loss: 'LOSS',
      dataIOText: 'DATAIO_TEXT',
      sampleText: 'SAMPLE_TEXT',
      sampleTable: 'SAMPLE_TABLE',
      intersection: 'INTERSECTION',
      'K-S': 'KS_EVALUATION',
      ROC: 'ROC_EVALUATION',
      Lift: 'LIFT_EVALUATION',
      Gain: 'GAIN_EVALUATION',
      Accuracy: 'ACCURACY_EVALUATION',
      RecallBinary: 'RECALL_BINARY_EVALUATION',
      PrecisionBinary: 'PRECISION_BINARY_EVALUATION',
      RecallMulti: 'RECALL_MULTI_EVALUATION',
      PrecisionMulti: 'PRECISION_MULTI_EVALUATION',
      Summary: 'EVALUATION_SUMMARY'
    },
    icons: {
      normal: {
        fullscreen: require('../../icons/jobdetail_outputsfromjob_visualization_fullscreen_default.png'),
        close: require('@/icons/jobdetail_outputsfromjob_visualization_close_default.png'),
        left: require('@/icons/jobdetail_outputsfromjob_visualization_pagebackward_default.png'),
        right: require('@/icons/jobdetail_outputsfromjob_visualization_pageforward_default.png'),
        success: require('@/icons/dashboard_job_complete.png'),
        failed: require('@/icons/dashboard_job_failed.png')
      },
      active: {
        fullscreen: require('@/icons/jobdetail_outputsfromjob_visualization_fullscreen_click.png'),
        close: require('@/icons/jobdetail_outputsfromjob_visualization_close_click.png'),
        left: require('@/icons/jobdetail_outputsfromjob_visualization_pagebackward_click.png'),
        right: require('@/icons/jobdetail_outputsfromjob_visualization_pageforward_click.png')
      },
      hover: {
        fullscreen: require('@/icons/jobdetail_outputsfromjob_visualization_fullscreen_hover.png'),
        close: require('@/icons/jobdetail_outputsfromjob_visualization_close_hover.png'),
        left: require('@/icons/jobdetail_outputsfromjob_visualization_pagebackward_hover.png'),
        right: require('@/icons/jobdetail_outputsfromjob_visualization_pageforward_hover.png')
      }
    },
    currentCvTab: 0,
    evaluationFlags: [],
    evaluationInstances: []
  },
  mutations: {
    TOGGLE_SIDEBAR: state => {
      if (state.sidebar.opened) {
        Cookies.set('sidebarStatus', 1)
      } else {
        Cookies.set('sidebarStatus', 0)
      }
      state.sidebar.opened = !state.sidebar.opened
      state.sidebar.withoutAnimation = false
    },
    CLOSE_SIDEBAR: (state, withoutAnimation) => {
      Cookies.set('sidebarStatus', 1)
      state.sidebar.opened = false
      state.sidebar.withoutAnimation = withoutAnimation
    },
    TOGGLE_DEVICE: (state, device) => {
      state.device = device
    },
    SWITCH_REQ_SIMULATE: (state, flag) => {
      state.isOpenReqSimulate = flag
      Cookies.set('isOpenReqSimulate', flag)
    },
    INIT_MODEL_OUTPUT: (state) => {
      state.currentCvTab = 0
      state.evaluationFlags = []
    },
    CHANGE_CV_TAB: (state, index) => {
      state.currentCvTab = index
    },
    SET_CV_FLAGS: (state, arr) => {
      state.evaluationFlags = arr
    },
    SET_CURVE_INSTANCES: (state, arr) => {
      state.evaluationInstances = arr
    }

  },
  actions: {
    ChangeCvTab({ commit }, index) {
      commit('CHANGE_CV_TAB', index)
    },
    SetCvFlags({ commit }, arr) {
      commit('SET_CV_FLAGS', arr)
    },
    SetCurveInstances({ commit }, arr) {
      commit('SET_CURVE_INSTANCES', arr)
    },
    InitModelOutput({ commit }) {
      commit('INIT_MODEL_OUTPUT')
    },
    CloseSideBar({ commit }, { withoutAnimation }) {
      commit('CLOSE_SIDEBAR', withoutAnimation)
    },
    ToggleDevice({ commit }, device) {
      commit('TOGGLE_DEVICE', device)
    },
    SwitchReqSimulate({ commit }, flag) {
      commit('SWITCH_REQ_SIMULATE', flag)
    }
  }
}

export default app
