The readme describes how to create and delete an EKS cluster and KFP services.

#### Creating EKS cluster

    export CLUSTER_NAME="torchx-dev"
    export EKS_VERSION="1.21"
    envsubst < torchx-dev-eks-template.yml > torchx-dev-eks.yml
    eksctl create cluster -f torchx-dev-eks.yml

See https://docs.aws.amazon.com/eks/latest/userguide/platform-versions.html for the latest EKS version

#### Creating KFP

    Source doc: https://www.kubeflow.org/docs/components/pipelines/legacy-v1/installation/standalone-deployment/#deploying-kubeflow-pipelines

    export PIPELINE_VERSION=1.8.1
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
    kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"

See https://github.com/kubeflow/pipelines/releases for the latest KFP version

#### Applying KFP role binding

    kubectl create namespace torchx-dev
    kubectl apply -f kfp_volcano_role_binding.yaml

#### Creating torchserve

    https://github.com/pytorch/serve/tree/master/kubernetes/EKS

#### Installing volcano

    kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

    Install `vcctl`

#### Starting etcd service

    kubectl apply -f etcd.yaml

#### Deleting KFP services

    cd torchx-dev-1-18 && kfctl delete -V -f torchx-dev-kfp.yml

#### Deleting EKS cluster

    eksctl delete -f torch-dev-eks.yml

This command most likely will fail. EKS uses CloudFormation to create many resources that
are hard to remove. If the command fails there needs to be manual cleanup:
* Clean up the associated VPC. Go to AWS Console -> VPC -> Press `Delete`. This will
point you the ENI and NAT that needs to be deleted manually.
* Clean up the CloudFormation template. Go to AWS Console -> CNF -> delete corresponding templates.

### Gotchas:

* The directory where `torchx-dev-kfp.yml` is located should be the same name as eks cluster

* The node groups in the EKS cluster HAVE to be spread to more than a single AZ, otherwise there
 will be problems with `istio-ingress`

* KFP troubleshooting: https://www.kubeflow.org/docs/components/pipelines/legacy-v1/troubleshooting/

* Enable Kubernetes nodes to access AWS account resources: https://stackoverflow.com/a/64617080/1446208

* Torchserve fails with `DownloadArchiveException` : https://github.com/pytorch/serve/issues/1218
