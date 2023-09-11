import argparse
import os
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, roc_curve
from fate_client.pipeline.utils.test_utils import JobConfig  # 假设这个库在你的环境中存在

def main(config="../../config.yaml", param="./lgb_breast_config.yaml"):

    # obtain config
    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    assert isinstance(param, dict)
    data_guest = param["data_guest"]
    data_host = param["data_host"]
    idx = param["idx"]
    label_name = param["label_name"]

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        print(f"config: {config}")
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    config_param = {
        "objective": "binary",
        "num_leaves": param["num_leaves"],
        "learning_rate": param["learning_rate"],
        "n_estimators": param["n_estimators"],
        "max_bin": param["max_bin"],
    }

    # prepare data
    df_guest = pd.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
    df_host = pd.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)
    df = df_guest.join(df_host, rsuffix="host")
    print('data shape is {}'.format(df.shape))
    y = df[label_name]
    X = df.drop(label_name, axis=1)

    x_train, x_test, y_train, y_test = X, X, y, y  # no split here

    # Train the model
    clf = lgb.LGBMClassifier(**config_param)
    clf.fit(x_train, y_train)

    # Prediction
    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]

    try:
        auc_score = roc_auc_score(y_test, y_prob)
    except BaseException:
        print("no auc score available")
        return

    recall = recall_score(y_test, y_pred, average="macro")
    pr = precision_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    ks = max(tpr - fpr)
    result = {"auc": auc_score, "recall": recall, "precision": pr, "accuracy": acc}
    print(result)
    return {}, result

if __name__ == "__main__":
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY LightGBM JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./lgb_breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)
