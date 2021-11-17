# LUS-COVID-CHALLENGE


## Introduction  

[Lung ultrasound (LUS) ](https://www.youtube.com/watch?v=_Q0cTG3ZlHk&amp;ab_channel=MedCram-MedicalLecturesExplainedCLEARLY)is a non-invasive, pragmatic and time-tested tool for evaluating and discriminating respiratory pathology at the bedside. Indeed, when LUS is interpreted by *expert* clinicians, its predictive accuracy can match CT scanners. Recently, ultrasound-on-a-chip technology has made LUS extremely portable ([pluggable into a mobile phone](https://www.butterflynetwork.com/uk/)) and cheap enough (2000USD vs 30,000USD+) for use in resource limited settings. This makes it particularly useful in COVID-19, where its portability enables decentralized respiratory evaluations (at home or in triage centers rather than in the hospital) and simple inter-patient disinfection.  However, while acquisition may be simple, interpretation is comparatively challenging, prone to subjective bias as well as a lack of standardization.  The challenge is to use deep learning methods to predict if a patient has COVID-19 based on the LUS images!


## Installation

```
pip install -e .
```

## Notebook

To get an idea of the data and how to use the dataloaders see the data loader example noteboook.

Some comments:

- Nb folds just means how the data is split for the data loaders, 5 folds == .8 / .2 train / test split
- The data loader includes data augmentation, see dataset.py in the deepchest folder.
- Additional preprocessing is done as seen in the preprocessing.py file. 
- Indices are saved in the model_saved folder.

```
config.preprocessing_train_eval = "independent_dropout(.2);"
```

E.g. this means with prob .2, each site is dropped for each patient.

## Google cloud compute 

### Spawn a new VM on Google Cloud Platform

- Make sure that the `lus-nfs` instance is running. It serves the NFS drives to store our data.
- Create an instance with a V100 GPU. Put your name in the instance, e.g. `jb-1`, you are responsible for stopping it and removing it when it is not used anymore.
```
INSTANCE_NAME="your-name-1"
gcloud compute instances start lus-nfs && \
gcloud beta compute \
    --project=jb-unsupervised instances create $INSTANCE_NAME \
    --zone=europe-west4-a \
    --machine-type=n1-standard-8 \
    --subnet=default \
    --network-tier=PREMIUM \
    --maintenance-policy=TERMINATE \
    --service-account=storage-full@jb-unsupervised.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/compute,https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=type=nvidia-tesla-v100,count=1 \
    --image-family cluster-template \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --reservation-affinity=any \
    --disk=auto-delete=no,name=imagenet-ssd,device-name=imagenet-ssd,mode=ro
```
- Connect to the machine `gcloud compute ssh $INSTANCE_NAME`.
- Create and use your own directory on the network drive in `/data`.
```bash
mkdir /data/YOUR_USERNAME
```
- To work on the code base, you can clone this repo and install it in the `--user` space.
```bash
# Install your own deepchest from your data directory on the drive
cd /data/YOUR_USERNAME
git clone https://github.com/epfl-iglobalhealth/LUS-COVID-CHALLENGE
cd LUS-COVID-CHALLENGE
pip install --user -e .
```
- You can stop this machine or delete it. `/data` is persistent.

### Edit files remotely

You can use [vscode](https://code.visualstudio.com/) editor and mount a remote directory with the [Remote Development extensions](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) to edit your files.

- Make sure that `lus-nfs` instance is running.
- On your laptop, run `gcloud compute config-ssh`.
- Make sure you can access the instance by running `ssh user_name_on_gcp@lus-nfs.europe-west4-a.jb-unsupervised`. You can find `user_name_on_gcp` by running `whoami` on any gcp machine.
- Connect to the host `user_name_on_gcp@lus-nfs.europe-west4-a.jb-unsupervised` in VScode and open the `/lus-disk/YOUR_USERNAME/LUS-COVID-CHALLENGE` directory.

Note that you are editing files on the `lus-nfs` instance which has no GPU. To run experiments, start a GPU machine and the `/data` should contain your latest changes in VScode.

## Links

- [Main deepchest repo](https://github.com/epfl-iglobalhealth/LUS-COVID-main)

- [Student deepchest paper (draft)](https://www.overleaf.com/project/61910ac2312c4addcf45741f)

