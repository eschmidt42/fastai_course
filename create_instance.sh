#!/usr/bin/env bash
# for all steps required to work with the comput engine see: https://course.fast.ai/start_gcp.html
# how to set up network permissions and edit the vm to enable browser interaction with jupyter notebook: https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/
# sshing into the vm: gcloud compute ssh --zone=europe-west1-b my-fastai-instance
# note: reading and writing between Google Cloud Storage and the compute engine requires permission. those can be found at the very bottom of the vm properties (when in the browser)
export IMAGE_FAMILY="pytorch-latest-gpu" # or "pytorch-latest-cpu" for non-GPU instances
export ZONE="europe-west1-b"
export INSTANCE_NAME="my-fastai-instance"
export INSTANCE_TYPE="n1-highmem-8" # budget: "n1-highmem-4"

# budget: 'type=nvidia-tesla-k80,count=1'
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-p100,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible
