# Input data (not distributed)

The pipeline expects the following layout. None of these files is included
in the repository.

```
datos/
  dfTwitter/                     290 tab-separated files (*.tab)
  df_icare_original.xlsx         iCARE survey microdata, Colombia
  diccionario de icare.xlsx      iCARE data dictionary
```

## Twitter/X corpus

Spanish-language COVID-19 tweets collected between 31 March and 18 August
2020, published by C. Salazar as "Affective analysis on Twitter Spanish
COVID19" (Kaggle):
https://www.kaggle.com/datasets/carlossalazar/affective-analysis-on-twitter-spanish-covid19

Download the dataset and place its `.tab` files, with their original names
(for example `Covid-19_Spa_5000_10Abril.tab`), in `datos/dfTwitter/`.

## iCARE survey

The iCARE microdata were accessed under a data agreement between the iCARE
study (Montreal Behavioural Medicine Centre) and Universidad EAFIT and cannot
be redistributed. Requests for access should be directed to the iCARE study:
https://icare.mbmc-cmcm.ca/

Without these two files, stage 06 (and the cross-source comparison that
depends on it) cannot be run; every Twitter/X stage runs on the public corpus
alone.

To read the data from another location, set the environment variable
`ICARE_DIR_DATOS` to the folder that contains `dfTwitter/`.
