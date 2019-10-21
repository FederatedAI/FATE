# coding=utf-8  
import os,sys
import json

partyid="10000"
exchangeip="127.0.0.1"
clustercommip="127.0.0.1"
fip="127.0.0.1"
rollip="127.0.0.1"
proxyip="127.0.0.1"

proxyport=9370
exchangeport=9370
clustercommport=9394
rollport=8011


def get_new_json(module_name,filepath):
    with open(filepath, 'rb') as f:
        json_data = json.load(f)
        a = json_data
        if module_name == "exchange":
            a["route_table"][partyid]={"default":[{"ip": proxyip,"port": proxyport}]}
        elif module_name == "proxy":
            a["route_table"]={"default":{"default":[{"ip": exchangeip,"port": exchangeport}]},\
            partyid:{"eggroll":[{"ip": clustercommip,"port": clustercommport}]}}
        elif module_name == "python":
            a["servers"]["roll"]={"host":rollip,"port": rollport}
            a["servers"]["clustercomm"]={"host":clustercommip,"port": clustercommport}
            a["servers"]["proxy"]={"host":proxyip,"port": proxyport}
        else:
            print("Please input right module name!")
            
    f.close()
    return json_data
    
def rewrite_json_file(filepath,json_data):
    with open(filepath, 'w') as f:
        json.dump(json_data,f,indent=4, separators=(',', ': '))
    f.close()
 
if __name__ == '__main__':
    
    module_name = sys.argv[1]
    json_path = sys.argv[2]
        
    m_json_data = get_new_json(module_name,json_path)    
    rewrite_json_file(json_path,m_json_data)
