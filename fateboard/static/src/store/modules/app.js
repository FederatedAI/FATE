import Cookies from 'js-cookie'

const app = {
  state: {
    sidebar: {
      opened: !+Cookies.get('sidebarStatus'),
      withoutAnimation: false
    },
    device: 'desktop',
    isOpenReqSimulate: Cookies.get('isOpenReqSimulate') !== 'false',
    projectType: [
      {
        value: 1,
        label: 'Insurance pricing'
      },
      {
        value: 2,
        label: 'Credit risk'
      }
    ],
    jobType: [
      {
        value: 1,
        label: 'intersection'
      },
      {
        value: 2,
        label: 'feature engineering'
      },
      {
        value: 3,
        label: 'model training'
      },
      {
        value: 4,
        label: 'model prdiction'
      }
    ]
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
    }
  },
  actions: {
    ToggleSideBar: ({ commit }) => {
      commit('TOGGLE_SIDEBAR')
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
