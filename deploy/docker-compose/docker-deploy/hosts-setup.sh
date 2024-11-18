#! /bin/bash

# Copyright 2019 The FATE Authors. All Rights Reserved.
#
### Group, user and folder creation
## Add  docker group if not exists
echo "Adding group 'docker'"
groupadd -f docker
##
echo "Add user 'fate'"
useradd -s /bin/bash -g docker -d /home/fate -m fate
## Set the password
echo "Set password for 'fate'"
passwd fate
mkdir -p /data/projects/fate
## change ownership
chown -R fate:docker /data/projects/fate

#### (Re)Install Docker with latest engine and compose plugin
## snap is not suitable since it requires docker-compose.yml file on  Home tree
## apt version does not support  composer plugin (may be for Ubuntu 20.04) but this causes denial of "docker compose" 
## form present in FATE scripts and allows only "docker-compose"

# remove the existing docker and compose
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do sudo apt-get remove $pkg; done
sudo snap remove docker 
#sudo snap remove docker --purge  //to save time by avoiding saving the snapshot


sudo apt-get update
sudo apt-get install ca-certificates curl gnupg

#install keys for repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

#add repository
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

#update and install the latest version
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
