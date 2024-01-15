# examples; you can adjust it if you need.

python sshe_lr_launcher.py --parties guest:9999 host:10000 --log_level INFO --guest_data ../data/breast_hetero_guest.csv --host_data ../data/breast_hetero_host.csv

python secureboost_launcher.py --parties guest:9999 host:10000 --log_level INFO

python sshe_nn_launcher.py --parties guest:9999 host:10000 --log_level INFO

python fedpass_nn_launcher.py --parties guest:9999 host:10000 --log_level INFO