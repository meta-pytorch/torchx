#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux
minikube delete
minikube start --driver=docker --cpus=max --memory=max --nodes=2

echo "=== Cluster status before registry addon ==="
kubectl get nodes
kubectl get pods -A

minikube addons enable registry || {
    echo "=== Registry addon timed out, waiting manually ==="
    kubectl get pods -n kube-system -l kubernetes.io/minikube-addons=registry -o wide || true
}

# Wait for registry pod with longer timeout (minikube's 6min timeout is often not enough)
# Only wait for the main registry deployment, not the registry-proxy DaemonSet
# which may have image pull issues in CI environments
echo "=== Waiting for registry deployment (up to 10 minutes) ==="
kubectl wait --for=condition=available deployment/registry \
    -n kube-system \
    --timeout=600s

echo "=== Registry pods ready ==="
kubectl get pods -n kube-system -l kubernetes.io/minikube-addons=registry -o wide

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
