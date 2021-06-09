import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

files = ['./hateful_memes_run_test-lr=1.79045221e-05.csv',
         './hateful_memes_run_test-lr=4.51652551e-03.csv',
         './hateful_memes_run_test-lr=5.5633e-04.csv',
         './hateful_memes_run_test-lr=5.70739300e-05.csv',
         './hateful_memes_run_test-lr=8.26783289e-04.csv'
]
for csv_file in files:
    print(f'Report for {csv_file}')
    result = pd.read_csv(csv_file)
    y_true = result['label']
    y_pred = result['predicted_label']
    target_names = ['non-hateful','hateful']
    print(f'classification report :\n{classification_report(y_true, y_pred, target_names=target_names)}')
    print(f'AUROC = {roc_auc_score(y_true, y_pred)}')
