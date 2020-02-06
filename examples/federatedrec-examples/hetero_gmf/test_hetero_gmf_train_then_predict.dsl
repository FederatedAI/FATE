{
  "components": {
    "dataio_0": {
      "module": "DataIO",
      "input": {
        "data": {
        "data": [
          "args.train_data"
        ]
        }
      },
      "output": {
        "data": [
        "train"
        ],
        "model": [
        "dataio"
        ]
      }
    },
    "hetero_gmf_0": {
      "module": "HeteroGMF",
      "input": {
        "data": {
        "train_data": [
          "dataio_0.train"
        ]
        }
      },
      "output": {
        "data": [
        "train"
        ],
        "model": [
        "hetero_gmf"
        ]
      }
    },
    "evaluation_0": {
      "module": "Evaluation",
      "input": {
        "data": {
          "data": ["hetero_gmf_0.train"]
        }
      },
      "output": {
        "data": ["evaluate"]
      },
      "need_deploy":true
    }
  }
}