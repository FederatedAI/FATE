import argparse
import json
import os

def set_config(guest_party_id, host_party_id):
    os.makedirs('./jobs', exist_ok=True)
    with open(os.path.join('./jobs', 'party_info.json'), 'w') as fw:
        json.dump({
            'guest': guest_party_id,
            'host': host_party_id
        }, fw)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("guest_party_id", type=int, help="please input guest party id")
    arg_parser.add_argument("host_party_id", type=int, help="please input host party id")
    args = arg_parser.parse_args()
    guest_party_id = args.guest_party_id
    host_party_id = args.host_party_id
    set_config(guest_party_id, host_party_id)

