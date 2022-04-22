import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from file_manager import FileManager

def main_pipeline():
    fm = FileManager(output_path='output_data')

    # load data
    data = pd.read_parquet('META FINAL DATA/gan_16.parquet')

    X = data.drop(['Approval', 'Race'], axis=1)
    y = data[['Approval']]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train
    model = XGBClassifier(random_state=42)
    model.fit(X_train.values, y_train.values)

    # get preds
    preds = pd.DataFrame(model.predict(X_test), columns=['target_pred'])

    y_train.rename(columns={0: 'target_train'})
    y_train.columns = y_train.columns.astype(str)
    y_test.rename(columns={0: 'target_test'})
    y_test.columns = y_test.columns.astype(str)

    model_stats = {'accuracy': accuracy_score(y_test, model.predict(X_test))}

    # write
    output_path = fm.get_modified_output_path()

    model.save_model(f'{output_path}/model.json')
    with open(f'{output_path}/accuracy.json', 'w') as fp:
        json.dump(model_stats, fp)

main_pipeline()