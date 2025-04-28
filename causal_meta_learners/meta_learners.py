from scipy.spatial.distance import cdist
from causalinference import CausalModel
import numpy as np
import matplotlib.pyplot as plt
from causal_meta_learners.classification_models import MLModel
from causal_meta_learners.survival_models import SurvivalModel



############################################################
########### Causal Inference Meta Learner Modesl ###########
############################################################
def train_t_learner(causal_data_dict, model_type_dict, task="classification", task_setup_dict={}, handle_imbalance=True, verbose=True):
    '''
    Train a T-Learner model using the provided causal data and model type dictionary.
    ITE = mu_1(X) - mu_0(X)
    '''

    model_name, hyperparams = list(model_type_dict.items())[0]

    t_learner_models = {}
    predictive_performance = {}
    for A_ in [0, 1]:
        X_train = causal_data_dict["X"]["train"][causal_data_dict["A"]["train"] == A_]
        X_test = causal_data_dict["X"]["test"][causal_data_dict["A"]["test"] == A_]
        Y_train = causal_data_dict["Y"]["train"][causal_data_dict["A"]["train"] == A_]
        Y_test = causal_data_dict["Y"]["test"][causal_data_dict["A"]["test"] == A_]
        if task == "classification":
            Y_train = Y_train[:, 1]
            Y_test = Y_test[:, 1]

        if verbose: print(f"(T-Learner) Training {model_name} for A={A_} ...")

        if task == "classification":
            if model_name == "FCN":
                hyperparams["input_size"] = X_train.shape[1]
                hyperparams["output_size"] = len(np.unique(Y_train))
            model = MLModel(model_name, hyperparams, handle_imbalance=handle_imbalance)
        elif task == "survival":
            model = SurvivalModel(model_name, hyperparams, 
                                  extrapolate_median=task_setup_dict.get("extrapolate_median", False), 
                                  random_seed=task_setup_dict.get("random_seed", 42))
        else:
            raise ValueError(f"Unsupported task type: {task}")
        
        model.fit(X_train, Y_train)
        report = model.evaluate(X_test, Y_test)
        if verbose: 
            if task=="classification": print(report)
            if task=="survival": print(f"C-index: {report:.4f}")
            print()

        t_learner_models[A_] = model
        predictive_performance[A_] = report


    if task == "classification":
        ITE = t_learner_models[1].predict(causal_data_dict["X"]["total"]) - t_learner_models[0].predict(causal_data_dict["X"]["total"])
    elif task == "survival":
        ITE = t_learner_models[1].predict_metric(causal_data_dict["X"]["total"], metric=task_setup_dict.get("metric", "median"), max_time=task_setup_dict.get("max_time", np.inf)) \
                - t_learner_models[0].predict_metric(causal_data_dict["X"]["total"], metric=task_setup_dict.get("metric", "median"), max_time=task_setup_dict.get("max_time", np.inf))
    ATE = np.mean(ITE)

    if verbose: print(f"Average Treatment Effect (ATE) for T-Learner: {ATE:.4f}")

    return {"ATE": ATE, "ITE": ITE, 'predictive_performance': predictive_performance}


def train_s_learner(causal_data_dict, model_type_dict, task="classification", task_setup_dict={}, handle_imbalance=True, verbose=True):
    '''
    Train an S-Learner model using the provided causal data and model type dictionary.
    ITE = mu(X, 1) - mu(X, 0)
    '''

    model_name, hyperparams = list(model_type_dict.items())[0]

    X_train = causal_data_dict["X"]["train"]
    X_test = causal_data_dict["X"]["test"]
    Y_train = causal_data_dict["Y"]["train"]
    Y_test = causal_data_dict["Y"]["test"]
    A_train = causal_data_dict["A"]["train"]
    A_test = causal_data_dict["A"]["test"]

    if task == "classification":
        Y_train = Y_train[:, 1]
        Y_test = Y_test[:, 1]

    X_A_train = np.concatenate([X_train, A_train.reshape(-1, 1)], axis=1)
    X_A_test = np.concatenate([X_test, A_test.reshape(-1, 1)], axis=1)

    if verbose: print(f"(S-Learner) Training {model_name} ...")
    
    if task == "survival":
        model = SurvivalModel(model_name, hyperparams, 
                              extrapolate_median=task_setup_dict.get("extrapolate_median", False), 
                              random_seed=task_setup_dict.get("random_seed", 42))
    elif task == "classification":
        if model_name == "FCN":
            hyperparams["input_size"] = X_A_train.shape[1]
            hyperparams["output_size"] = len(np.unique(Y_train))
        model = MLModel(model_name, hyperparams, handle_imbalance=handle_imbalance)
    else:
        raise ValueError(f"Unsupported task type: {task}")
    
    model.fit(X_A_train, Y_train)
    report = model.evaluate(X_A_test, Y_test)
    predictive_performance = {0: report}
    if verbose: 
        if task=="classification": print(report)
        if task=="survival": print(f"C-index: {report:.4f}")
        print()

    X_total_treatment = np.concatenate([causal_data_dict["X"]["total"], np.ones((len(causal_data_dict["X"]["total"]),  1))], axis=1)
    X_total_control   = np.concatenate([causal_data_dict["X"]["total"], np.zeros((len(causal_data_dict["X"]["total"]), 1))], axis=1)

    if task == "classification":
        ITE = model.predict(X_total_treatment) - model.predict(X_total_control)
    elif task == "survival":
        ITE = model.predict_metric(X_total_treatment, metric=task_setup_dict.get("metric", "median"), max_time=task_setup_dict.get("max_time", np.inf)) \
                - model.predict_metric(X_total_control, metric=task_setup_dict.get("metric", "median"), max_time=task_setup_dict.get("max_time", np.inf))
    ATE = np.mean(ITE)

    if verbose: print(f"Average Treatment Effect (ATE) for S-Learner: {ATE:.4f}")

    return {"ATE": ATE, "ITE": ITE, 'predictive_performance': predictive_performance}


def train_x_learner(causal_data_dict, model_type_dict, task="classification", task_setup_dict={}, handle_imbalance=True, verbose=True):
    '''
    Train an X-Learner model using the provided causal data and model type dictionary.
    Step (1): Train two T-Learner models for A=0 and A=1
    Step (2a): tau_0 = mu_1(X_i) - Y_i(0) for Control group (A=0) and tau_1 = Y_i(1) - mu_0(X_i) for Treatment group (A=1)
    Step (2b): Train the propensity score model using the entire dataset (g(X_i))
    Step (3): Calculate the Individual Treatment Effect (ITE) for the entire dataset ITE = g(X_i)*tau_0(X_i) + (1-g(X_i))*tau_1(X_i)
    '''

    model_name, hyperparams = list(model_type_dict.items())[0]

    # Step 1: Train two T-Learner models for A=0 and A=1
    t_learner_models = {}
    for A_ in [0, 1]:
        X_train = causal_data_dict["X"]["train"][causal_data_dict["A"]["train"] == A_]
        X_test = causal_data_dict["X"]["test"][causal_data_dict["A"]["test"] == A_]
        Y_train = causal_data_dict["Y"]["train"][causal_data_dict["A"]["train"] == A_]
        Y_test  = causal_data_dict["Y"]["test"][causal_data_dict["A"]["test"] == A_]

        if task == "classification":
            Y_train = Y_train[:, 1]
            Y_test = Y_test[:, 1]

        if verbose: print(f"(X-Learner-Step 1 (mu_{A_})) Training {model_name} ...")
        
        if task == "classification":
            if model_name == "FCN":
                hyperparams["input_size"] = X_train.shape[1]
                hyperparams["output_size"] = len(np.unique(Y_train))
            model = MLModel(model_name, hyperparams, handle_imbalance=handle_imbalance)
        elif task == "survival":
            model = SurvivalModel(model_name, hyperparams, 
                                  extrapolate_median=task_setup_dict.get("extrapolate_median", False), 
                                  random_seed=task_setup_dict.get("random_seed", 42))
        else:
            raise ValueError(f"Unsupported task type: {task}")
        
        model.fit(X_train, Y_train)
        report = model.evaluate(X_test, Y_test)
        if verbose: 
            if task=="classification": print(report)
            if task=="survival": print(f"C-index: {report:.4f}")
            print()

        t_learner_models[A_] = model

    # Step 2a: Calculate Individual Treatment Effect (ITE) Estimates
    x_learner_models = {}
    inverse_label_mapping_dict = {}
    for A_ in [0, 1]:
        X_train = causal_data_dict["X"]["train"][causal_data_dict["A"]["train"] == A_]
        X_test = causal_data_dict["X"]["test"][causal_data_dict["A"]["test"] == A_]

        if verbose: print(f"(X-Learner-Step 2 (tau_{A_})) Training {model_name} ...")

        if task == "classification":
            Y_train = (causal_data_dict["Y"]["train"][causal_data_dict["A"]["train"] == A_][:, 1] - t_learner_models[1-A_].predict(X_train)) * (2*A_-1)
            Y_test  = (causal_data_dict["Y"]["test"][causal_data_dict["A"]["test"] == A_][:, 1]   - t_learner_models[1-A_].predict(X_test))  * (2*A_-1)
            
            # If Y_train contains negative labels, map them to non-negative integers
            unique_labels = np.unique(Y_train)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            inverse_label_mapping_dict[A_] = {v: k for k, v in label_mapping.items()}

            Y_train = np.array([label_mapping[label] for label in Y_train])
            Y_test = np.array([label_mapping[label] for label in Y_test])

            if model_name == "FCN":
                hyperparams["input_size"] = X_train.shape[1]
                hyperparams["output_size"] = len(np.unique(Y_train))
            model = MLModel(model_name, hyperparams, handle_imbalance=handle_imbalance)

        elif task == "survival":
            model = SurvivalModel(model_name, hyperparams, 
                                  extrapolate_median=task_setup_dict.get("extrapolate_median", False), 
                                  random_seed=task_setup_dict.get("random_seed", 42))
            raise NotImplementedError("Survival task not supported in Step 2a of X-Learner")
            
        model.fit(X_train, Y_train)
        report = model.evaluate(X_test, Y_test)
        if verbose:
            if task=="classification": print(report)
            if task=="survival": print(f"C-index: {report:.4f}")
            print()

        x_learner_models[A_] = model

    # Step 2b: Train the propensity score model using the entire dataset
    X_total = causal_data_dict["X"]["total"]
    A_total = causal_data_dict["A"]["total"]
    propensity_model = MLModel('LogisticRegression', {}, handle_imbalance=False)
    propensity_model.fit(X_total, A_total)

    # Step 3: Calculate the Individual Treatment Effect (ITE) for the entire dataset
    if task == "classification":
        tau_predictions = {A_: np.array([inverse_label_mapping_dict[A_][pred] for pred in x_learner_models[A_].predict(X_total)]) for A_ in [0, 1]}
        ITE = propensity_model.predict(X_total) * tau_predictions[0] + (1 - propensity_model.predict(X_total)) * tau_predictions[1]
    elif task == "survival":
        raise NotImplementedError("Survival task not supported in Step 3 of X-Learner")
    ATE = np.mean(ITE)

    if verbose: print(f"Average Treatment Effect (ATE) for X-Learner: {ATE:.4f}")

    return {"ATE": ATE, "ITE": ITE}



############################################################
################# Matching ATE Estimator ###################
############################################################
def train_matching_learner(causal_data_dict, model_type_dict, task="classification", task_setup_dict={}, verbose=True):

    model_name, hyperparams = list(model_type_dict.items())[0]

    num_matches = hyperparams.get("num_matches", 5)
    distance_metric = hyperparams.get("distance_metric", "euclidean")

    X_total = causal_data_dict["X"]["total"]
    A_total = causal_data_dict["A"]["total"]
    Y_total = causal_data_dict["Y"]["total"]

    if task == "classification":
        Y_true = Y_total[:, 1]
        Predicted_Y_total = None
        predictive_performance = {0: None}

    elif task == "survival":
        Y_true = Y_total[:, 0] # use true event times for survival task (Not Used in Matching!)
        model = SurvivalModel(model_name, hyperparams, 
                              extrapolate_median=task_setup_dict.get("extrapolate_median", False), 
                              random_seed=task_setup_dict.get("random_seed", 42))
        
        X_A_total = np.concatenate([X_total, A_total.reshape(-1, 1)], axis=1)
        model.fit(X_A_total, Y_total)
        report = model.evaluate(X_A_total, Y_total)
        predictive_performance = {0: report}
        if verbose: print(f"C-index: {report:.4f}")
        if verbose: print()

        Predicted_Y_total = model.predict_metric(X_A_total, metric=task_setup_dict.get("metric", "median"), 
                                                 max_time=task_setup_dict.get("max_time", np.inf))
        
    else:
        raise ValueError(f"Unsupported task type: {task}")
    
    ITE = matching_with_k_nearest_neighbors(X_total, A_total, Y_true, Predicted_Y_total, task_type=task, 
                                            num_matches=num_matches, metric=distance_metric)
    ATE = np.mean(ITE)

    if verbose: print(f"Average Treatment Effect (ATE) for Matching Learner: {ATE:.4f}")

    return {"ATE": ATE, "ITE": ITE, 'predictive_performance': predictive_performance}



def matching_with_k_nearest_neighbors(X_total, A_total, True_Y_total, Predicted_Y_total, task_type='survival', num_matches=5, metric='euclidean'):
    

    # Precompute distances between all points in X_total
    distances = cdist(X_total, X_total, metric=metric)

    # Initialize the result array
    nearest_indices = np.zeros((X_total.shape[0], num_matches), dtype=int)

    for i in range(X_total.shape[0]):
        # Mask for rows with a different A_total value
        mask = A_total != A_total[i]
        
        # Get the indices and distances of points with different A_total
        filtered_indices = np.where(mask)[0]
        filtered_distances = distances[i, filtered_indices]
        
        # Find the indices of the nearest neighbors within the filtered distances
        nearest_neighbors = np.argsort(filtered_distances)[:num_matches]
        
        # Map back to the original indices
        nearest_indices[i] = filtered_indices[nearest_neighbors]

    if task_type == 'survival':
        psi_hat_i = Predicted_Y_total - np.mean([Predicted_Y_total[nearest_neighbors_i] for nearest_neighbors_i in nearest_indices], axis=1)
    elif task_type == 'classification':
        psi_hat_i = True_Y_total - np.mean([True_Y_total[nearest_neighbors_i] for nearest_neighbors_i in nearest_indices], axis=1)
    else:
        raise ValueError("task_type must be either 'survival' or 'classification'")
    
    psi_hat_i *= (2 * A_total - 1)  # Multiply by 1 for originally treated, -1 for originally control

    return psi_hat_i