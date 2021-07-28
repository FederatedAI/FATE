import os
import json
import sys
import re
import requests
import traceback
import datetime
from deal_rollsite_audit_log_settings import LOG_INDEX, ELASTIC_SEARCH_URL, ELASTIC_SEARCH_AUTH, ELASTIC_SEARCH_USER, ELASTIC_SEARCH_PASSWORD, HOST_ROLE_PARTY_ID

LOG_PATH = ""
SERVER_IP = None
EXCHANGE_TYPE = "general"


def run():
    progress = read_progress()
    end_pos = progress.get("end_pos", -1)
    last_st_ino = progress.get("st_ino", -1)
    last_st_mtime = progress.get("st_mtime", -1)
    now_st_ino = os.stat(LOG_PATH).st_ino
    print(f"last inode num: {last_st_ino}, mtime: {last_st_mtime}")
    print(f"{LOG_PATH} inode num is {now_st_ino}")
    if last_st_ino != -1 and last_st_ino != now_st_ino:
        # create time not match, log path have change
        print(f"last inode num is {last_st_ino}, but now is {now_st_ino}, log path have change, search all pending log")
        last_deal_log_path, pending_paths = search_pending_logs(os.path.dirname(LOG_PATH), last_st_ino, last_st_mtime)
        print(f"find last deal log path: {last_deal_log_path}")
        print(f"find pending paths: {pending_paths}")
        deal_log(last_deal_log_path, end_pos)
        for pending_path in pending_paths:
            deal_log(pending_path, -1)
        # reset end pos
        end_pos = -1
    end_pos = deal_log(LOG_PATH, end_pos)
    progress["end_pos"] = end_pos
    progress["st_mtime"] = os.stat(LOG_PATH).st_mtime
    progress["st_ino"] = os.stat(LOG_PATH).st_ino
    save_progress(progress)


def deal_log(LOG_PATH, end_pos):
    if not LOG_PATH:
        return end_pos
    audit_logs = []
    with open(LOG_PATH) as fr:
        line_count = get_file_line_count(fr)
        print(f"{LOG_PATH} end pos: {end_pos}, line count: {line_count}")
        if line_count > end_pos + 1:
            fr.seek(end_pos + 1)
            while True:
                line = fr.readline()
                if not line:
                    break
                audit_log = deal_line(line)
                merge_audit_log(audit_log, audit_logs)
            end_pos = fr.tell()
        else:
            print(f"{LOG_PATH} no change")
    if audit_logs:
        bulk_save(audit_logs)
    return end_pos


def merge_audit_log(audit_log, audit_logs):
    if audit_log:
        #audit_logs.append(json.dumps({"index": {"_index": "fate_rollsite_exchange_audit"}}))
        audit_logs.append('{"index":{}}')
        audit_logs.append(json.dumps(audit_log))


def search_pending_logs(log_dir, st_ino, st_mtime):
    last_deal_log_path = None
    pending_paths = []

    year_dirs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    year_dirs.sort(key=lambda f: os.stat(f).st_mtime, reverse=True)
    for year_dir in year_dirs:
        print(f"search year dir: {year_dir}")
        month_dirs = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if os.path.isdir(os.path.join(year_dir, f))]
        month_dirs.sort(key=lambda f: os.stat(f).st_mtime, reverse=True)
        year_search = False
        for month_dir in month_dirs:
            print(f"search month dir: {month_dir}")
            day_dirs = [os.path.join(month_dir, f) for f in os.listdir(month_dir) if os.path.isdir(os.path.join(month_dir, f))]
            day_dirs.sort(key=lambda f: os.stat(f).st_mtime, reverse=True)
            month_search = False
            for day_dir in day_dirs:
                print(f"search day dir: {day_dir}")
                last_deal_log_path, day_pending_paths = get_pending_logs(day_dir, st_ino, st_mtime)
                if day_pending_paths:
                    print(f"get pending path: {day_pending_paths}")
                    pending_paths.extend(day_pending_paths)
                else:
                    print(f"{day_dir} no pending path, break")
                    break
            else:
                # all day dir have pending_paths
                month_search = True
            if not month_search:
                break
        else:
            # all day dir have pending_paths
            year_search = True
        if not year_search:
            break
    return last_deal_log_path, pending_paths


def get_pending_logs(day_dir, st_ino, st_mtime):
    pending_paths = []
    st_mtime_match_path = None
    for f in os.listdir(day_dir):
        f_p = os.path.join(day_dir, f)
        if os.path.isfile(f_p) and f.startswith("rollsite-audit.log") and os.stat(f_p).st_mtime >= st_mtime:
            if os.stat(f_p).st_ino == st_ino:
                st_mtime_match_path = f_p
            else:
                pending_paths.append(f_p)
    return st_mtime_match_path, pending_paths


def get_file_line_count(fp):
    fp.seek(0, 2)
    return fp.tell()


def progress_file_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "deal_rollsite_log_progress.json")


def read_progress():
    p_p = progress_file_path()
    if not os.path.exists(p_p):
        return {}
    with open(p_p) as fr:
        return json.load(fr)


def save_progress(progress):
    p_p = progress_file_path()
    with open(p_p, "w") as fw:
        json.dump(progress, fw, indent=4)


def deal_line(src):
    #a = "[INFO ][36165610][2021-03-19 20:08:05,935][grpc-server-9370-30,pid:32590,tid:89][audit:87] - task={taskId=202103192007180194594}|src={name=202103192007180194594,partyId=9999,role=fateflow,callback={ip=127.0.0.1,port=9360}}|dst={name=202103192007180194594,partyId=10000,role=fateflow}|command={name=/v1/party/202103192007180194594/arbiter/10000/clean}|operator=POST|conf={overallTimeout=30000}"
    meta_data = {}
    try:
        split_items = src.split(" - ")
        meta_line = split_items[1].strip()
        meta_data["logTime"] = re.findall("\[.*?\]", split_items[0])[2].strip("[").strip("]")
        meta_data["logTime"] = (datetime.datetime.strptime(meta_data["logTime"], "%Y-%m-%d %H:%M:%S,%f") - datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        for meta_item_str in meta_line.split("|"):
            meta_item_key = meta_item_str[:meta_item_str.index("=")]
            meta_item_value_str = meta_item_str[meta_item_str.index("=") + 1:]
            if meta_item_value_str.find("{") == 0:
                meta_item_value = str_to_dict(meta_item_value_str[1:-1])
            else:
                meta_item_value = meta_item_value_str
            meta_data[meta_item_key] = meta_item_value
        meta_data["jobId"] = meta_data["task"]["taskId"]
        meta_data["jobDate"] = (datetime.datetime.strptime(meta_data["jobId"][:14], "%Y%m%d%H%M%S") - datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        meta_data["server"] = SERVER_IP
        meta_data["exchangeType"] = EXCHANGE_TYPE
        meta_data["src"]["role"] = "host" if meta_data["src"]["partyId"] in HOST_ROLE_PARTY_ID else "guest"
        meta_data["dst"]["role"] = "host" if meta_data["dst"]["partyId"] in HOST_ROLE_PARTY_ID else "guest"
    except Exception as e:
        traceback.print_exc()
    return meta_data


def str_to_dict(src):
    key = ""
    value = ""
    current = 1  # 1 for key, 2 for value

    d = {}
    i = 0
    while True:
        c = src[i]
        if c == "{":
            j = i + 1
            sub_str = ""
            while True:
                if src[j] == "}":
                    j = j + 1
                    break
                else:
                    sub_str += src[j]
                    j = j + 1
            sub = str_to_dict(sub_str)
            if current == 2:
                d[key] = sub
            i = j
        else:
            if c == "=":
                current = 2
            elif c == ",":
                d[key] = value
                key = ""
                value = ""
                current = 1
            else:
                if current == 1:
                    key += c
                elif current == 2:
                    value += c
            i = i + 1
        if i == len(src):
            if key and value:
                d[key] = value
            break
    return d


def upload(audit_log):
    res = requests.post("/".join([ELASTIC_SEARCH_URL, LOG_INDEX, "_doc"]), json=audit_log)
    print(res.json())


def bulk_save(audit_logs):
    data = "\n".join(audit_logs) + "\n"
    if ELASTIC_SEARCH_AUTH:
        res = requests.post("/".join([ELASTIC_SEARCH_URL, LOG_INDEX, "_doc", "_bulk"]),
                            data=data,
                            headers={'content-type':'application/json', 'charset':'UTF-8'},
                            timeout=(30, 300),
                            auth=(ELASTIC_SEARCH_USER, ELASTIC_SEARCH_PASSWORD))
    else:
        res = requests.post("/".join([ELASTIC_SEARCH_URL, LOG_INDEX, "_doc", "_bulk"]),
                            data=data,
                            headers={'content-type':'application/json', 'charset':'UTF-8'},
                            timeout=(30, 300))
    print(res.text)
    print(res.json())


if __name__ == "__main__":
    LOG_PATH = sys.argv[1]
    SERVER_IP = sys.argv[2]
    EXCHANGE_TYPE = sys.argv[3]
    run()