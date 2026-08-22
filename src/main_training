import os
import joblib
import argparse
import datetime
import numpy as np
import pandas as pd
from src.eval_utils import *
from src.helper_utils import *
from src.data_utils import *
from src.trainer import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  help="path to the dataset and preprocessing config file", default="configs/config.yml")
    parser.add_argument("--trainer", help="path to trainer and evaluation config file", default="configs/trainer_config.yml")
    args = parser.parse_args()
   
    config =  load_config(args.config)
    trainer = load_config(args.trainer)

    RANDOM_SEED = config['random_seed']
    np.random.seed(RANDOM_SEED)
    
    X_train, y_train, X_val, y_val = load_data(config)
    X_train_scaled, X_val_scaled = scale_data(X_train, X_val, config['preprocessing']['scaler_type'])

    if config['balancing']['do_balance']: # balance data
        X_train_scaled , y_train = balance_data_transformation(X_train_scaled, y_train, balance_type= config['balancing']['method'], sampling_strategy=config['balancing']['sampling_strategy'], k=5,  random_state=RANDOM_SEED)
        X_train = X_train_scaled.copy()
    
    model_comparison = {} # model comparison stats dictionary
    models = {} # trained models dictionary

    # results path for trained models and evaluation figures
    now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    path = 'models/{}/'.format(now)
    if not os.path.exists(path):
        os.makedirs(path)

    if trainer['trainer']['Random_forest']['train']:
       models["Random_forest"] = train_random_forest(X_train, y_train, X_val, y_val, RANDOM_SEED , model_comparison, trainer) # Random forest does not need scaled data

    if trainer['trainer']['Logistic_Regression']['train']:
        models['Logistic_Regression'] = train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val, RANDOM_SEED, model_comparison, trainer)

    if trainer['trainer']['Neural_Network']['train']:
        models['Neural_Network'] = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val, RANDOM_SEED, model_comparison, trainer)    

    if trainer['trainer']['KNN']['train']:
        models['KNN'] = train_knn(X_train_scaled, y_train, X_val_scaled, y_val, RANDOM_SEED, model_comparison, trainer)

    if trainer['trainer']['Voting_Classifier']['train']:
        models['Voting_Classifier'] = train_voting_classifier(X_train, y_train, X_val, y_val, models, RANDOM_SEED, model_comparison, trainer)

    # Save the models
    model_path = path + "trained_models.pkl"
    joblib.dump(models, model_path)
    print('Model saved at: {}'.format(model_path))
    print('Evaluation plots saved at: {}evaluation/plot'.format(path))

    if  model_comparison:
        # Save the model comparison
        model_comparison_path = path + "model_comparison-(validation dataset).png"
        save_model_comparison(model_comparison, model_comparison_path)

        print('\nModels comparison:\n')
        print(pd.DataFrame(model_comparison).T.to_markdown())