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
    "homo_nn_0": {
      "module": "HomoNN",
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
          "homo_nn"
        ]
      }
    },
    "homo_nn_1": {
      "module": "HomoNN",
      "input": {
        "data": {
          "eval_data": [
            "dataio_0.train"
          ]
        },
        "model": ["homo_nn_0.homo_nn"]
      },
      "output": {
        "data": [
          "train2"
        ],
        "model": [
          "homo_nn2"
        ]
      }
    }
  }
}
