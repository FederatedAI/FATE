# coding=utf-8  
import os,sys
import json

partyId="10000"
exchangeip="0.0.0.0"
fip="0.0.0.0"

flip="0.0.0.0"
fbip="0.0.0.0"

rip="0.0.0.0"
pip="0.0.0.0"
sip1="0.0.0.0"
sip2="0.0.0.0"


def get_new_json(module_name,filepath):
	with open(filepath, 'rb') as f:
		json_data = json.load(f)
		a = json_data
		if module_name == "exchange":
			a["route_table"][partyId]={"default":[{"ip": pip,"port": 9370}]}
		elif module_name == "proxy":
			a["route_table"]={"default":{"default":[{"ip": exchangeip,"port": 9370}]},\
			partyId:{"fate":[{"ip": fip,"port": 9394}],"fateflow":[{"ip": flip,"port": 9360}]}}

			if sip1 == "":
				print("[INFO] The number of serving is 0")
			else:
				if sip2 == "":
					a["route_table"][partyId]["serving"]=[{"ip": sip1,"port": 8001}]
				else:
					a["route_table"][partyId]["serving"]=[{"ip": sip1,"port": 8001},{"ip": sip2,"port": 8001}]
		elif module_name == "python":
			a["servers"]["roll"]={"host":rip,"port": 8011}
			a["servers"]["federation"]={"host":fip,"port": 9394}
			a["servers"]["proxy"]={"host":pip,"port": 9370}
			
			a["servers"]["fateboard"]={"host":fbip,"port": 8080}
			a["servers"]["fateflow"]={"host":flip,"grpc.port": 9360,"http.port": 9380}
			

			if sip1 == "":
				print("[INFO] The number of serving is 0")
			else:
				if sip2 == "":
					a["servers"]["servings"]=["{}:8001".format(sip1)]
				else:
					a["servers"]["servings"]=["{}:8001".format(sip1),"{}:8001".format(sip2)]
			a["party_id"]=partyId
		else:
			print("[ERROR] Please input right module name!")
			
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
