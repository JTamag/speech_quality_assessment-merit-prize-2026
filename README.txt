VoiceMOS 2022 / BVCC — student assignment pack
==============================================

Give students this folder (or a zip of it). The assignment brief is:

  Assignment_Brief.pdf

What is included
----------------
data/train.csv              labeled training utterances (2254 rows, 224 acoustic features)
data/dev.csv                labeled development utterances (387 rows)
data/test.csv               labeled test utterances (741 rows) for local scoring
data/test_unlabeled.csv     same test rows without MOS/class columns (optional blind protocol)
data/feature_description.csv
data/dataset_paths.yaml     local and public dataset locations
scripts/preprocess.py       train-only median impute + StandardScaler
scripts/train_svm_baseline.py
requirements.txt
citations.bib
docs/citations_crosschecked.csv

Instructor-only
---------------
instructor/test_labels.csv
instructor/baseline_results.json   (created after you run the baseline)
instructor/INSTRUCTOR_NOTES.txt

How students run the baseline
-----------------------------
  python -m pip install -r requirements.txt
  cd scripts
  python train_svm_baseline.py --data-dir ../data --out-dir ../outputs_student

Do not use system_id, utterance_id, or split as classifier inputs.
Fit imputation and scaling on train only. Tune on dev. Evaluate test once.

Audio is not required to run the SVM. Waveforms live at the path in
data/dataset_paths.yaml if a later task needs them.
