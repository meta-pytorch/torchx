#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux
minikube delete
minikube start --driver=docker --cpus=max --memory=max --nodes=2
minikube addons enable registry --wait=600

# setup multi node volumes
# https://github.com/kubernetes/minikube/issues/12360#issuecomment-1430243861
minikube addons disable storage-provisioner
minikube addons disable default-storageclass
minikube addons enable volumesnapshots
minikube addons enable csi-hostpath-driver
kubectl patch storageclass csi-hostpath-sc -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

# install volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.7.0/installer/volcano-development.yaml

# create namespace
kubectl create namespace torchx-dev

# portforwarding
kubectl port-forward --namespace kube-system service/registry 5000:80 &
