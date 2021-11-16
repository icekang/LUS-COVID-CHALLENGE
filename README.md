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

