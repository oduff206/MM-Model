## Data & Models
Raw data, processed CSVs, and trained models are intentionally excluded from this repository.

To run the project:
1. Go to the Kaggle NCAA March Madness dataset:

https://www.kaggle.com/competitions/march-machine-learning-mania-2025/overview

Download four files:
- MNCAATourneyCompactResults.csv
- MNCAATourneySeeds.csv
- MRegularSeasonCompactResults.csv
- MTeams.csv

3. Place the CSVs in `data/raw/`.
4. Run notebooks in order:
   - `01_build_tables.ipynb`
   - `02_build_training_set.ipynb`
   - `03_train_model.ipynb`

To make prediction: edit predict.py, and at very bottom change:
- Season: enter year (2023 for example)
- Team A ID: enter team ID (check MTeams.csv for reference)
- Team B ID: enter team ID (check MTeams.csv for reference)

Run 'python predict.py' in terminal
   
 ``` 
   mm_model/
├── backend/
│   └── predict.py            # Inference script for matchup predictions
│
├── notebooks/
│   ├── 01_build_tables.ipynb        # Raw data → team-season feature tables
│   ├── 02_build_training_set.ipynb  # Features → ML training dataset
│   └── 03_train_model.ipynb         # Model training & evaluation
│
├── data/
│   ├── raw/                  # Kaggle CSVs (ignored by Git)
│   └── processed/            # Derived tables (generated, ignored by Git)
│
├── models/                   # Trained models (generated, ignored by Git)
│
├── .gitignore                # Excludes data, models, env files
```
