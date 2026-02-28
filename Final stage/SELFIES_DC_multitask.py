import shutil
from dotenv import load_dotenv
import os

from pandas.tests.tools.test_to_datetime import epochs

from Code.Full_systems.Selfies_Mol.Multitask_Classifier.Utils.random_combinations import \
    create_hyperparameter_combinations
from Code.Utils.util_methods import NNUtils
import pandas as pd
import deepchem as dc
import logging
import numpy as np
from rdkit import Chem, DataStructs
import selfies as sf
from tqdm.notebook import tqdm
import deepchem as dc
import json
from Code.Full_systems.Selfies_Mol.Multitask_Classifier.Utils.progress_callback	import ProgressBarCallback
from statistics import mean, stdev
from datetime import datetime
import torch
import gc

print(f"Current working directory: {os.getcwd()}")

base = NNUtils.find_project_root(os.getcwd())
print(f"Project root found: {base}")

load_dotenv(f'{base}/.env')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



def DC_multitask_fit(X_train, y_train, selfies_group_dict, model_name_=None, model_name_extension=None, nb_epochs=50, batch_size=50, layer_sizes=[3000,1000], learning_rate=0.001, dropouts=0.0, subset_size=1.0, debug=False):
    '''
    :param X_train: Training data (Pandas DF)
    :param y_train: Training labels (Pandas DF)
    :param selfies_group_dict: Dictionnary containing the indexing of de SELFIES parts the keys are the SELFIES parts and the values are the indexes
    :param model_name_: Name of the model to fit. If none, it will be retrieved from the .env file
    :param model_name_extension: Extension of the model name (e.g. model_X_100, None by default meaning no extension)
    :param nb_epochs: Number of epochs to train the model
    :param batch_size: Size of the batches to train.
    :param layer_sizes: The size of each hidden layer, provided in a list.
    :param learning_rate: Learning rate during the training.
    :param dropouts: Dropouts rate during training.
    :param subset_size: Size of the subset to train on (1 by default)
    :param debug: If debug information should be printed
    :return: Boolean if the model already existed before
    '''

    model_name = os.getenv("MODEL_NAME") if model_name_ is None else model_name_
    model_name_full = f"{model_name}" if model_name_extension is None else f"{model_name}_{model_name_extension}"
    model_directory = f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/{model_name}/{model_name_full}'
    model_path = f'{model_directory}/{model_name_full}_model'
    model_summary_path = f'{model_directory}/{model_name_full}_summary.txt'
    model_parameters_path = f'{model_directory}/{model_name_full}_parameters.json'

    paths = {
        'model_directory': model_directory,
        'model_path': model_path,
        'model_summary_path': model_summary_path,
        'model_parameters_path': model_parameters_path
    }

    if not os.path.exists(f'{model_path}/checkpoint1.pt') or not os.path.exists(f'{model_directory}/.done'):

        if debug:
            print(f'The trained model will be saved to {model_directory}')
            print(f'Model name: {model_name}')

        #os.makedirs(model_directory, exist_ok=True)

        # Remove the directory and its contents if exists, so it can be overwritten
        if os.path.exists(model_directory):
            shutil.rmtree(model_directory)

        # Recreate the empty directory
        os.makedirs(model_directory)

        if debug: print(f'The size of the training set is: {X_train.shape[0]}')

        # transform into deepchem datasets
        train_dataset = dc.data.NumpyDataset(X_train, y_train)
        if debug: print('DeepChem datasets created')

        #with open(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/selfies_group_dict.json', 'r') as json_file:
        #    selfies_group_dict = json.load(json_file)
        
        with open(f'{model_directory}/selfies_group_dict_{model_name}.json', 'w') as json_file:
            json.dump(selfies_group_dict, json_file, indent=4)

        
        with open(model_summary_path, 'w') as file:  # 'w' will overwrite the file if it exists
            # write text into the file
            file.write(f"Model {model_name_full} \n\n")
            file.write(f"Model path: {model_path}\n")
            file.write('--------------------------------------------------\n')
            file.write(f"Training epochs: {nb_epochs}\n")
            file.write(f"Training set size: {X_train.shape[0]}\n")
            file.write(f"Subset size: {int(subset_size * 100)}%\n")
            file.write('--------------------------------------------------\n')
            file.write(f'Tuned hyperparameters:\n')
            file.write('--------------------------------------------------\n')
            file.write(f"Batch size: {batch_size}\n")
            file.write(f"Layer sizes: {layer_sizes}\n")
            file.write(f"Learning rate: {learning_rate}\n")
            file.write(f"Dropouts: {dropouts}\n")
            file.write('--------------------------------------------------\n')
        
        model_parameters = {
            'epochs': nb_epochs,
            'n_tasks': y_train.shape[1],
            'n_features': int(os.getenv('MAX_MASS')),
            'layer_sizes': layer_sizes,
            'dropouts': dropouts,
            'learning_rate': learning_rate,
            'batch_size' : batch_size,
            'model_path' : model_path
        }

        with open(model_parameters_path, 'w') as json_file:
            json.dump(model_parameters, json_file, indent=4)

        if debug: print(f"Parameters saved to {model_parameters_path}")

        model = dc.models.MultitaskClassifier(
            n_tasks=y_train.shape[1],
            n_features=int(os.getenv('MAX_MASS')),
            layer_sizes=layer_sizes,
            dropouts=dropouts,
            learning_rate=learning_rate,
            batch_size = batch_size,
            model_dir = model_path
        )
        steps_per_epoch = X_train.shape[0] // batch_size
        progress_bar_callback = ProgressBarCallback(steps_per_epoch, nb_epochs, return_epochs=False, model_name=model_name_full)

        average_loss = model.fit(train_dataset, nb_epoch=nb_epochs, callbacks=[progress_bar_callback])
        

        progress_bar_callback.close()

        #model.save(model_path)
        if debug: print(f'Model saved to {model_path}')

        with open(model_summary_path, 'a') as file: 
            # write text into the file
            file.write(f"Average loss: {average_loss}\n")
        
        existed_before = False

        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    else:
        print(f'Model {model_name_full} already exists at {model_path}')
        existed_before = True
    
    return existed_before

def DC_multitask_predict(X_test, y_test, model_path=None, model_parameters_path=None):
    """
    Predict the labels of the test dataset using the model.
    :param X_test: Test data
    :param y_test: Test labels
    :param model_path: Path to the model
    :param model_parameters_path: Path to the model parameters
    :return: Predictions of the model
    """

    with open(model_parameters_path, 'r') as json_file:
        model_parameters = json.load(json_file)

    model = dc.models.MultitaskClassifier(
        n_tasks=model_parameters['n_tasks'],
        n_features=model_parameters['n_features'],
        layer_sizes=model_parameters['layer_sizes'],
        dropouts=model_parameters['dropouts'],
        learning_rate=model_parameters['learning_rate'],
        batch_size = model_parameters['batch_size'],
        model_dir = model_path
    )
    model.restore()

    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    predictions = model.predict(test_dataset)

    return predictions


def eval_results(predictions, test_dataset, model_summary_path=None, encoding_length=None, print_results=True, debug=False):
    '''
    Evaluate the performance of the model on the test dataset.
    :param predictions: Predictions of the model
    :param test_dataset: The test dataset
    :param model_summary_path: Path to the model summary file, to save the results
    :param encoding_length: The length of the encoding of a molecule. If none, it will be retrieved from the .env file
    :param print_results: If the results should be printed
    :param debug: If debug information should be printed
    :returns: True positive and false positive metrics in % such as: {'true_positives': {'avg','stdev','cv'},'false_positives': {'avg','stdev','cv'}}
    '''

    # NOT CORRECT YET!!
    # THE [nop] SHOULD NOT BE INCLUDED IN THE EVALUATION !!! (not an issue anymore, since it is a bitwise comparison)

    if encoding_length is None: encoding_length = int(os.getenv("ENCODING_BITS"))*int(os.getenv("MAX_SELFIES_LENGTH"))

    tp=[] # true positive
    fn=[] # false negative
    fp=[] # false positive
    pr=[] # total number of predicted bits
    tp_p=[] # true pos %
    fp_p=[] # false pos %

    cutoff = 0.5    # predicted >= 0.5 will turn bit=1

    for q in tqdm(range(len(test_dataset)), desc='Loop over all test molecules', unit='molecule'):   # loop over all test molecules

        # get predicted fingerprint of molecule q
        pred = []
        for i in predictions[q]:
            if i[1] >= cutoff:
                pred.append(1)
            else:
                pred.append(0)

        # get real fingerprint of molecule q
        real = test_dataset.y[q]

        bit = 0
        a=0
        b=0
        c=0
        d=0
        e=0

        for i in range(encoding_length):
            if real[i]==1 and pred[i]==1:     # true pos (correct prediction)
                a=a+1
            if real[i]==1 and pred[i]==0:     # false neg (missed)
                b=b+1
            if real[i]==0 and pred[i]==1:     # false pos (not correct)
                c=c+1
            if real[i]==1: # count number of 'on-bits'
                d=d+1
            if pred[i]==1: # count number of predicted 'on-bits'
                e=e+1
        
        epsilon = 10e-7
        
        tp.append(a)  # true pos
        fn.append(b)  # false neg
        fp.append(c)  # false pos
        pr.append(e)  # number of predicted on-bits
        fp_p.append(int(c/(e+epsilon)*100)) # false pos / predicted on-bits * 100%
        tp_p.append(int(a/(d+epsilon)*100)) # true pos / real number on-bits * 100%

    # % True positive average, stdev and cv% for all test molecules
    epsilon = 10e-7
    tp_avg = int (mean(tp_p))
    tp_sd = int (stdev(tp_p))
    tp_cv = int (tp_sd/(tp_avg+epsilon)*100)
    if print_results:
        print (f'BITWISE EVALUATION OF TEST_DATASET CONTAINING: {len(test_dataset)} MOLECULES')
        print (f'--------------------------------------------------------------------')
        print (f'TRUE POS:    AVG={tp_avg}%    STDEV={tp_sd}    CV%={tp_cv}')

    # % False positive average, stdev and cv% for all test molecules
    fp_avg = int (mean(fp_p))
    fp_sd = int (stdev(fp_p))
    fp_cv = int (fp_sd/(fp_avg+epsilon)*100)

    if print_results:
        print (f'FALSE POS:   AVG={fp_avg}%    STDEV={fp_sd}    CV%={fp_cv}')

    if model_summary_path is not None:
        with open(model_summary_path, 'a') as file:
            file.write(f'\n\n\n')
            file.write(f'{datetime.now()}\n\n')
            file.write(f'BITWISE EVALUATION OF TEST_DATASET CONTAINING: {len(test_dataset)} MOLECULES\n')
            file.write(f'--------------------------------------------------------------------\n')
            file.write(f'TRUE POS:    AVG={tp_avg}%    STDEV={tp_sd}    CV%={tp_cv}\n')
            file.write(f'FALSE POS:   AVG={fp_avg}%    STDEV={fp_sd}    CV%={fp_cv}\n')
    
    res = {
        'true_positives': {
            'avg': -1,
            'stdev': -1,
            'cv': -1
        },
        'false_positives': {
            'avg': -1,
            'stdev': -1,
            'cv': -1
        }
    }

    return res


def eval_model(predictions, selfies_group_dict, test_dataset, selfies_X_test, threshold=0.5, model_name=None, model_folder=None, model_summary_path=None, debug=False):
    '''
    Evaluate the model based on the predictions and the test dataset.
    :param predictions: Predictions of the model
    :param selfies_group_dict: Dictionnary containing the indexing of SELFIES parts
    :param test_dataset: The test dataset
    :param selfies_X_test: The SELFIES of the test dataset
    :param threshold: The threshold to consider a prediction as correct
    :param model_name: Name of the model
    :param model_name_extension: Extension of the model name (e.g. model_X_100, None by default meaning no extension)
    :param model_folder: Folder where the model is saved
    :param model_summary_path: Path to the model summary file
    :param debug: If debug information should be printed
    :return: The simiarity metrics of the predicted molecules such as: {
        'prediction_simiarity_rates': {
            'correct': int,
            '100%+': int,
            '90%+': int,
            '60%+': int
        },
        'prediction_similarity_values': {
            'correct': int,
            '100%+': int,
            '90%+': int,
            '60%+': int
        },
        'test_size': int,
        'fingerprint_similarity': float,
        'SELFIES_similarity': float
    }
    '''

    # create a dictionairy (itos) to transfer the hot-encoding array into a array of SELFIES and SMILES (transposed of the selfies_group_dict)
    itos={}
    c=0
    for i in selfies_group_dict:
        itos[selfies_group_dict[i]]=i

    # Evaluation whole test set
    # check if predicted smiles == real smiles

    cutoff = threshold
    hit = 0
    score=[]
    to_print = []

    columns = ['SMILES_Index', 'Original_SMILES', 'Predicted_SMILES', 'Original_SELFIES', 'Predicted_SELFIES', 'Fingerprint_SIMILARITY', 'SELFIES_SIMILARITY']
    res_df = pd.DataFrame(columns=columns)

    epsilon = 10e-7

    for test_compound_id in tqdm(range(len(test_dataset)), desc='Evaluate all test molecules', unit='molecule'):   # loop over all test molecules
        # create hot-encoding array of molecule id
        pred = []
        # thresholding the predictions
        for i in predictions[test_compound_id]:
            if i[1] >= cutoff: #index 1 is for the porbability of the class being 1 (index 0 is the probability of the class being 0)
                pred.append(1)
            else:
                pred.append(0)

        predicted_selfies =''
        a = len(pred)
        b = len(itos)
        c = int (a/(b+epsilon))

        # iterate through the indexes of the prediction (one prediction is the one hot encoding of a SELFIES part)
        for i in range(c):
            # iterate through the indexes of the itos dictionary (length of a one hot encoding)
            for q in range(b):
                if pred[i*b+q]==1 and itos[q]!='[nop]':
                    # print (itos[q])
                    predicted_selfies = predicted_selfies + (itos[q])
                    # go to the encoding ot the next SELFIES part after the first 1 has been found for the one hot encoding
                    #break
        predicted_smiles = sf.decoder(predicted_selfies)

        # real molecule
        #compound_id = test_dataset.ids[test_compound_id]
        real_selfies = selfies_X_test.loc[test_compound_id]
        original_smiles = sf.decoder(real_selfies) # inferred from selfies

            # Convert SMILES to RDKit molecule object
        mol_a = Chem.MolFromSmiles(predicted_smiles)
        mol_b = Chem.MolFromSmiles(original_smiles)
        
        # Only proceed if both molecules are valid
        if mol_a is not None and mol_b is not None:
            a = Chem.RDKFingerprint(mol_a)
            b = Chem.RDKFingerprint(mol_b)
            score.append(DataStructs.FingerprintSimilarity(a, b, metric=DataStructs.DiceSimilarity))
            #to_print.append(f'{test_compound_id} ------- {predicted_smiles} -------- {original_smiles} {DataStructs.FingerprintSimilarity(a,b, metric=DataStructs.DiceSimilarity)}')
            fingerprint_similarity = DataStructs.FingerprintSimilarity(a,b, metric=DataStructs.DiceSimilarity)
            selfies_similarity = NNUtils.selfies_similarity(real_selfies, predicted_selfies)
            res_df.loc[test_compound_id] = [test_compound_id, original_smiles, predicted_smiles, real_selfies, predicted_selfies, fingerprint_similarity, selfies_similarity]

            
            if predicted_smiles == original_smiles:
                hit = hit + 1
            
        else:
            #print("Invalid molecule found, skipping.")
            pass
    
    avg_fingerprint_similarity = float(res_df['Fingerprint_SIMILARITY'].mean(skipna=True))
    avg_selfies_similarity = float(res_df['SELFIES_SIMILARITY'].mean(skipna=True))
    
    if model_name is not None and model_folder is not None:
        filename = f"{model_folder}/{model_name}_predictions.csv"
        res_df.to_csv(filename)

        if debug: print(f'The predictions for model {model_name} have been saved to {filename}')
    

    score = np.array(score)
    sum_score_1 = sum(score>=1)
    sum_score_09 = sum(score>=0.9)
    sum_score_06 = sum(score>=0.6)

    res = {
        'prediction_simiarity_rates': {
            'correct': hit/len(test_dataset.X)*100,
            '100%+': sum_score_1/len(test_dataset.X)*100,
            '90%+': sum_score_09/len(test_dataset.X)*100,
            '60%+': sum_score_06/len(test_dataset.X)*100
        },
        'prediction_similarity_values': {
            'correct': hit,
            '100%+': sum_score_1,
            '90%+': sum_score_09,
            '60%+': sum_score_06
        },
        'test_size': len(test_dataset.X),
        'fingerprint_similarity': avg_fingerprint_similarity,
        'SELFIES_similarity': avg_selfies_similarity
    }

    if model_summary_path is not None:
        with open(model_summary_path, 'a') as file:
            file.write(f'\n\n')
            file.write(f'{datetime.now()}\n')
            file.write(f'Correct smiles predictions: {hit} (={int(hit/len(test_dataset.X)*100)}%). Test set contains in total {len(test_dataset.X)} compounds.\n')
            file.write(f'Tanimoto similarity >= 1.0: {sum_score_1} (={int(sum_score_1/len(test_dataset.X)*100)}%). Test set contains in total {len(test_dataset.X)} compounds.\n')
            file.write(f'Tanimoto similarity >= 0.9: {sum_score_09} (={int(sum_score_09/len(test_dataset.X)*100)}%). Test set contains in total {len(test_dataset.X)} compounds.\n')
            file.write(f'Tanimoto similarity >= 0.6: {sum_score_06} (={int(sum_score_06/len(test_dataset.X)*100)}%). Test set contains in total {len(test_dataset.X)} compounds.\n')
            file.write('The Tanimoto similarity is calculated using the RDKit Fingerprint and the Dice similarity metric.\n')

    return res


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):  # convert numpy integers
        return int(obj)
    elif isinstance(obj, np.floating):  # convert numpy floats
        return float(obj)
    elif isinstance(obj, np.ndarray):  # convert numpy arrays
        return obj.tolist()
    else:
        return obj  # return as-is if already serializable  


def DC_multitask_evaluate(X_test, y_test, selfies_X_test, model_name_=None, model_name_extension=None, encoding_length=None, debug=False):
    '''
    Evaluate the model on the test dataset.
    :param X_test: Test data
    :param y_test: Test labels
    :param selfies_X_test: SELFIES of the test data
    :param model_name_: Name of the model to evaluate. If none, it will be retrieved from the .env file
    :param model_name_extension: Extension of the model name (e.g. model_X_100, None by default meaning no extension)
    :param encoding_length: Length of the encoding of a molecule. If none, it will be retrieved from the .env file
    :param debug: If debug information should be printed
    :return: The percentage of correct molecule predictions
    '''

    model_name = os.getenv("MODEL_NAME") if model_name_ is None else model_name_
    model_name_full = f"{model_name}" if model_name_extension is None else f"{model_name}_{model_name_extension}"
    model_directory = f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/{model_name}/{model_name_full}'
    model_path = f'{model_directory}/{model_name_full}_model'
    model_summary_path = f'{model_directory}/{model_name_full}_summary.txt'
    model_parameters_path = f'{model_directory}/{model_name_full}_parameters.json'
    model_evaluation_path = f'{model_directory}/{model_name_full}_evaluation.json'
    selfies_group_dict_path = f'{model_directory}/selfies_group_dict_{model_name}.json'

    with open(selfies_group_dict_path, 'r') as json_file:
        selfies_group_dict = json.load(json_file)

    if encoding_length is None: encoding_length = int(os.getenv("ENCODING_BITS"))*int(os.getenv("MAX_SELFIES_LENGTH"))

    with open(model_parameters_path, 'r') as json_file:
        model_parameters = json.load(json_file)

    model = dc.models.MultitaskClassifier(
        n_tasks=model_parameters['n_tasks'],
        n_features=model_parameters['n_features'],
        layer_sizes=model_parameters['layer_sizes'],
        dropouts=model_parameters['dropouts'],
        learning_rate=model_parameters['learning_rate'],
        batch_size = model_parameters['batch_size'],
        model_dir = model_path
    )
    model.restore()

    test_dataset = dc.data.NumpyDataset(X_test, y_test)

    predictions = model.predict(test_dataset)

    results = {}

    results = {
        'true_positives': {
            'avg': -1,
            'stdev': -1,
            'cv': -1
        },
        'false_positives': {
            'avg': -1,
            'stdev': -1,
            'cv': -1
        },
        'prediction_rates': {
            'correct': -1,
            '100%+': -1,
            '90%+': -1,
            '60%+': -1
        },
        'prediction_values': {
            'correct': -1,
            '100%+': -1,
            '90%+': -1,
            '60%+': -1
        },
        'test_size': -1,
        'fingerprint_similarity': -0.1,
        'SELFIES_similarity': -0.1
    }

    tp_fp_metrics = eval_results(predictions, test_dataset, model_summary_path=model_summary_path, encoding_length=encoding_length, debug=debug, print_results=debug)

    results['true_positives'] = tp_fp_metrics['true_positives']
    results['false_positives'] = tp_fp_metrics['false_positives']


    pred_similarity = eval_model(predictions, selfies_group_dict, test_dataset, selfies_X_test, model_name=model_name_full, model_folder=model_directory, model_summary_path=model_summary_path, debug=debug)

    results['prediction_rates'] = pred_similarity['prediction_simiarity_rates']
    results['prediction_values'] = pred_similarity['prediction_similarity_values']
    results['test_size'] = pred_similarity['test_size']
    results['fingerprint_similarity'] = float(pred_similarity['fingerprint_similarity'])
    results['SELFIES_similarity'] = float(pred_similarity['SELFIES_similarity'])

    ser_res = convert_to_serializable(results)

    with open(model_evaluation_path, "w") as json_file:
        json.dump(ser_res, json_file, indent=4)


    with open(f'{model_directory}/.done', 'w') as binary_file:
        # Write binary data to the file
        binary_file.write('This model has been successfully trained and evaluated.\nThis file is a marker for it. \nDO NOT DELETE!')

    return results['prediction_rates']['correct']


def DC_multitask_optimizer(rounds=50, training_epochs=50, model_name_=None, start_round=1, manual_combinations=None, debug=False):
    '''
    Optimize the model using a random search approach. The length of the encoding is determined by the length of the SELFIES parts dictionary * the max selfies lenght from the .env file.
    :param rounds: Number of rounds to optimize the model
    :param training_epochs: Number of epochs to train the model
    :param model_name_: Name of the model to optimize. If none, it will be retrieved from the .env file
    :param start_round: The round to start the optimization
    :param manual_combinations: Manually specify the combinations to use (as first combinations) (list of dictionaries)
    :param debug: If debug information should be printed
    :return:
    '''

    base_model_name = os.getenv("MODEL_NAME") if model_name_ is None else model_name_

    # Load the data
    X_train = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("X_TRAIN")}')#.loc[:30]
    X_test = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("X_TEST")}')#.loc[:30]
    y_train = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("Y_TRAIN")}')#.loc[:30]
    y_test = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("Y_TEST")}')#.loc[:30]
    selfies_X_test = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/selfies_X_test.pkl')#.loc[:30]

    with open(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/selfies_group_dict.json', 'r') as json_file:
        selfies_group_dict = json.load(json_file)

    best_correct_pred = 0
    best_correct_pred_round = -1

    hyperparameters = create_hyperparameter_combinations(rounds, manual_combinations=manual_combinations, learning_rate_= 0.001, randomize=False)#, nb_hidden_layers_= 2, hidden_layers_= [3000, 2000], learning_rate_= 0.001 , batch_size_= 100, dropout_= 0)

    # create a csv that contains the hyperparameters and the results (if not exists)
    if not os.path.exists(f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/tuning.csv'):
        with open(f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/tuning.csv', 'w') as file:
            file.write('round;model_full_name;layer_sizes;learning_rate;batch_size;dropout;correct_pred;start_timestamp;end_timestamp\n')

    for i in tqdm(range(len(hyperparameters)), desc='Optimizing model', unit='model'):

        start_time = datetime.now()

        model_name_extension = f"100"
        model_name = f"{base_model_name}_{i+start_round+1}"

        layer_sizes = hyperparameters[i]['layer_sizes']
        learning_rate = hyperparameters[i]['learning_rate']
        batch_size = hyperparameters[i]['batch_size']
        dropouts = hyperparameters[i]['dropout']

        if debug: print(f'Training model {model_name}')

        # Train the model:
        existed_before = DC_multitask_fit(X_train, y_train, selfies_group_dict, model_name_=model_name, model_name_extension=model_name_extension, nb_epochs=training_epochs, layer_sizes=layer_sizes, learning_rate=learning_rate, batch_size=batch_size, dropouts=dropouts, debug=debug)

        if not existed_before:

            if debug: print(f"Evaluating the predictions for model {model_name}")

            # Evaluate the predictions:
            correct_pred = DC_multitask_evaluate(X_test, y_test, selfies_X_test, model_name_=model_name, model_name_extension=model_name_extension, encoding_length=len(selfies_group_dict)*int(os.getenv("MAX_SELFIES_LENGTH")), debug=debug)

            if correct_pred > best_correct_pred:
                best_correct_pred = correct_pred
                best_correct_pred_round = i

            #print(f'The best correct prediction percentage so far is {best_correct_pred} in round {best_correct_pred_round}')

            with open(f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/tuning.csv', 'a') as file:
                file.write(f'{i+start_round};{model_name}_{model_name_extension};{layer_sizes};{learning_rate};{batch_size};{dropouts};{correct_pred};{start_time};{datetime.now()}\n')


# def DC_multitask_stratified_optimizer(max_rounds=50, training_epochs=50, model_name_=None, wait_rounds=6, strategy='coordinate descent', debug=False):
#     base_model_name = os.getenv("MODEL_NAME") if model_name_ is None else model_name_
#
#     # Load the data
#     X_train = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("X_TRAIN")}')  # .loc[:30]
#     X_test = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("X_TEST")}')  # .loc[:30]
#     y_train = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("Y_TRAIN")}')  # .loc[:30]
#     y_test = pd.read_pickle(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/{os.getenv("Y_TEST")}')  # .loc[:30]
#     selfies_X_test = pd.read_pickle( f'{base}/Code/Full_systems/Selfies_Mol/Featurization/selfies_X_test.pkl')  # .loc[:30]
#
#     with open(f'{base}/Code/Full_systems/Selfies_Mol/Featurization/selfies_group_dict.json', 'r') as json_file:
#         selfies_group_dict = json.load(json_file)
#
#     tuning_results_path = f'{base}/Code/Full_systems/Selfies_Mol/Multitask_Classifier/Models/tuning.csv'
#
#     if not os.path.exists(tuning_results_path):
#         with open(tuning_results_path, 'w') as file:
#             file.write('round;model_full_name;layer_sizes;learning_rate;batch_size;dropout;correct_pred;start_timestamp;end_timestamp\n')
#
#     if strategy == 'coordinate descent':
#         DC_multitask_coordinate_descent_optimizer(
#             X_train, X_test, y_train, y_test, selfies_X_test, selfies_group_dict,
#             tuning_results_path=tuning_results_path,
#             training_epochs=50,
#             base_model_name=base_model_name,
#             patience=6,
#             debug=debug
#         )
#
#
#
# def train_and_evaluate_DC_multitask(X_train, X_test, y_train, y_test, selfies_X_test, selfies_group_dict, tuning_results_path, full_name, hyperparameters, training_epochs=50, debug=False):
#
#     correct_preds = {} # key: model name, value: correct pred
#
#     for i in tqdm(range(len(hyperparameters)), desc='Optimizing model', unit='model'):
#         start_time = datetime.now()
#
#         model_name = full_name
#
#         layer_sizes = hyperparameters[i]['layer_sizes']
#         learning_rate = hyperparameters[i]['learning_rate']
#         batch_size = hyperparameters[i]['batch_size']
#         dropouts = hyperparameters[i]['dropout']
#
#         if debug: print(f'Training model {model_name}')
#
#         # Train the model:
#         existed_before = DC_multitask_fit(X_train, y_train, selfies_group_dict, model_name_=model_name, nb_epochs=training_epochs,
#                                           layer_sizes=layer_sizes, learning_rate=learning_rate, batch_size=batch_size,
#                                           dropouts=dropouts, debug=debug)
#
#         if not existed_before:
#
#             if debug: print(f"Evaluating the predictions for model {model_name}")
#
#             # Evaluate the predictions:
#             correct_pred = DC_multitask_evaluate(X_test, y_test, selfies_X_test, model_name_=model_name, encoding_length=None,
#                                                  debug=debug)
#
#             correct_preds[f'{model_name}'] = correct_pred
#
#             with open(tuning_results_path, 'a') as file:
#                 file.write(
#                     f'{i + start_round};{model_name}_{model_name_extension};{layer_sizes};{learning_rate};{batch_size};{dropouts};{correct_pred};{start_time};{datetime.now()}\n')
#
#         else:
#             training_results = pd.read_csv(tuning_results_path)
#             correct_pred = training_results.loc[training_results['model_full_name']==model_name, 'correct_pred']
#             correct_preds[model_name] = correct_pred
#
#     return correct_preds
#
#
# def DC_multitask_coordinate_descent_optimizer(X_train, X_test, y_train, y_test, selfies_X_test, selfies_group_dict, tuning_results_path, previous_hyperparameters, training_epochs=50, base_model_name=None, min_models=6, patience=3, opt_step=0 ,best_model_name=None, best_SMILES_similarity=0, fixed_hyperparameters={'nb_hidden_layers': -1, 'hidden_layers': -1, 'learning_rate': -1.0, 'batch_size': -1, 'dropout': -1.0}, debug=False):
#
#     current_level = opt_step
#     version = 0 # nb of currently explored parallel branch from the left
#     next_SMILES_similarity = 0
#     step_models = {} # models in this step (key: model name, value: SMILES accuracy)
#     fixed_hyps = [] # list of fixed hyperparameters in this level, so it doesn't fix twice the same
#     current_hyperparameters = []
#
#     hyps = [hyp for hyp, value in fixed_hyperparameters.items() if(value in (-1, -1.0))]
#
#     nb_possible_children = len(hyps)-1 # number of possible future explorations
#
#     model_name = f'{model_base_name}_optlvl{current_level}'
#
#     if current_level == 0:
#         existing_measurements = pd.read_csv(tuning_results_path)
#
#         # train enough models if there are not enough
#         if existing_measurements.shape[0] < min_models:
#             training_results = pd.read_csv(tuning_results_path)
#
#             # generate hyperparameters for the remaining
#             print('Training the initial models')
#             for i in tqdm(range(min_models-existing_measurements.shape[0]), desc='Training initial models', unit='model'):
#
#                 model_full_name = f'{model_name}_ver{version}_no{existing_measurements.shape[0]+i+1}_100'
#
#                 hyperparameters = create_hyperparameter_combinations(1, nb_hidden_layers_=fixed_hyperparameters['nb_hidden_layers'], hidden_layers_=fixed_hyperparameters['hidden_layers'], learning_rate_=fixed_hyperparameters['learning_rate'], batch_size_=fixed_hyperparameters['batch_size'], dropout_=fixed_hyperparameters['dropout'], progressbar=debug)
#
#                 train_and_evaluate_DC_multitask(X_train, X_test, y_train, y_test, selfies_X_test, selfies_group_dict, tuning_results_path, base_model_name, hyperparameters, training_epochs=training_epochs, debug=debug)
#
#         # find the best model
#         best_model_row = existing_measurements.loc[existing_measurements['correct_pred'].idxmax()]
#         best_model_name = best_model_row['model_full_name']
#         best_SMILES_similarity = best_model_row['correct_pred']
#

#         # TODO the selection of fixed hyperparameters need to be moved here, so the fixed hyperparameters are given to the recursion for the next step
#         # TODO recursive call here, one for each fixed hyperparameter ##################
#
#     else:
#         # it is not the base level, but at least deepness one
#
#         # select a hyperparameter to freeze
#         hyps = [hyp for hyp, value in fixed_hyperparameters.items() if (value in (-1, -1.0) and hyp not in fixed_hyps)]
#
#         if len(hyps) == 0: # the optimization is over
#             return True
#
#         # TODO: Remove the random Hyperparameter creation from here
#         # select a random hyperparameter to freeze
#         frozen_hyperparameter = random.choice(hyps)
#         fixed_hyps.append(frozen_hyperparameter)
#         frozen_hyp_value = previous_hyperparameters[frozen_hyperparameter]
#         new_fixed_hyperparameters = fixed_hyperparameters.copy()
#         new_fixed_hyperparameters[frozen_hyperparameter] = frozen_hyp_value
#
#         if frozen_hyperparameter == 'nb_hidden_layers':
#             new_fixed_hyperparameters['hidden_layers'] = [None]
#             fixed_hyps.append('hidden_layers')
#         elif frozen_hyperparameter == 'hidden_layers':
#             new_fixed_hyperparameters['nb_hidden_layers'] = len(new_fixed_hyperparameters['hidden_layers'])
#             fixed_hyps.append('nb_hidden_layers')
#
#         # create models with the selected hyperparameters
#         step_models[version] = {}
#         max_SMILES_similarity = 0
#         max_model = None
#         for i in tqdm(range(patience), desc=f'Training models for level {current_level}, version {version}', unit='model'):
#
#             model_full_name = f'{model_name}_ver{version}_no{i+1}_100'
#
#             done = False
#             c=0
#             while done == False:
#                 hyperparameters = create_hyperparameter_combinations(1, nb_hidden_layers_=new_fixed_hyperparameters['nb_hidden_layers'], hidden_layers_=new_fixed_hyperparameters['hidden_layers'], learning_rate_=new_fixed_hyperparameters['learning_rate'], batch_size_=new_fixed_hyperparameters['batch_size'], dropout_=new_fixed_hyperparameters['dropout'], progressbar=debug)
#                 if hyperparameters != previous_hyperparameters and hyperparameters not in current_hyperparameters:
#                     done = True
#                     current_hyperparameters.append(hyperparameters)
#                 else:
#                     c+=1
#                     if c==5: continue
#
#             correct_pred = train_and_evaluate_DC_multitask(X_train, X_test, y_train, y_test, selfies_X_test, selfies_group_dict,
#                                             tuning_results_path, base_model_name, hyperparameters,
#                                             training_epochs=training_epochs, debug=debug)[model_full_name]
#             step_models[version][model_full_name] = correct_pred
#
#             if max_SMILES_similarity < correct_pred:
#                 max_SMILES_similarity = correct_pred
#                 max_model = model_full_name
#
#         if max_SMILES_similarity > best_SMILES_similarity:
#             # TODO the selection of fixed hyperparameters need to be moved here, so the fixed hyperparameters are given to the recursion for the next step
#             # TODO recursive call here, one for each fixed hyperparameter ##################
#             pass
#         else:
#             # the rest of the branch should not be explored
#             return False




if __name__ == '__main__':
    DC_multitask_optimizer(rounds=50, training_epochs=1, model_name_=None, start_round=0)