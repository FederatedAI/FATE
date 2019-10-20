import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="path to log file")
    parser.add_argument("--output_dir", type=str, default="formatted_logs", help="path to output file")
    opt = parser.parse_args()
    log_file_name = os.path.basename(opt.log)
    if not os.path.exists(opt.log):
        raise FileNotFoundError("wrong log file path")
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    output = open(os.path.join(opt.output_dir, log_file_name.replace(".log", ".csv")), 'w')
    header = ["train_loss", "aggr_test_loss", "aggr_test_map", "aggr_test_recall", "server_test_loss",
              "server_test_map", "server_test_recall"]
    round_, train_loss, aggr_test_loss, aggr_test_map, aggr_test_recall, server_test_loss, server_test_map, server_test_recall = [
        list() for _ in range(8)]
    log_file = open(opt.log).readlines()
    for line in log_file:
        line = line.strip()
        if "Round" in line:
            round_.append(int(line.split(" ")[-2]))
        elif "aggr_train_loss" in line:
            train_loss.append(round(float(line.split(" ")[-1]), 4))
        elif "aggr_test_loss" in line:
            aggr_test_loss.append(round(float(line.split(" ")[-1]), 4))
        elif "aggr_test_map" in line:
            aggr_test_map.append(round(float(line.split(" ")[-1]), 4))
        elif "aggr_test_recall" in line:
            aggr_test_recall.append(round(float(line.split(" ")[-1]), 4))
        elif "server_test_loss" in line:
            server_test_loss.append(round(float(line.split(" ")[-1]), 4))
        elif "server_test_map" in line:
            server_test_map.append(round(float(line.split(" ")[-1]), 4))
        elif "server_test_recall" in line:
            server_test_recall.append(round(float(line.split(" ")[-1]), 4))
    output.write("round,train_loss,test_map,test_recall\n")
    for r, loss, mAP, recall in zip(round_, train_loss, server_test_map, server_test_recall):
        output.write("{},{},{},{}\n".format(r, loss, mAP, recall))
