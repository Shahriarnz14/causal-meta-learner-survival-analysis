from causalinference import CausalModel
import numpy as np
import matplotlib.pyplot as plt
from causal_meta_learners.meta_learners import train_t_learner, train_s_learner, train_x_learner, train_matching_learner


############################################################
############ Causal Inference Estimator Model ##############
############################################################
def get_meta_learner_results(causal_data_dict, ML_model, handle_imbalance, 
                             causal_methods, num_matches_list,
                             experiment_num, experiment_type,
                             task="classification", task_setup_dict={},
                             is_trimmed=False,
                             plot_results=True, print_results=True, verbose_output=False):
    results = {}
    for causal_method in causal_methods:
        if causal_method != "matching":
            print(f"Running {causal_method} ...")
            results[causal_method] = get_causal_estimate(causal_data_dict, ML_model, causal_method, task, task_setup_dict, handle_imbalance, verbose_output)
            if print_results: print(f"{causal_method} \t | \t ATE: {results[causal_method]['ATE']:.4f}")
        else:
            results[causal_method] = {}
            for num_matches in num_matches_list:
                print(f"Running {causal_method} (K={num_matches}) ...")
                
                if not ML_model:  # Check if ML_model is empty (e.g. for classification task with matching)
                    ML_model_dict = {causal_method: {"num_matches": num_matches}}
                else:  # If ML_model is not empty
                    ML_model_dict = {k: {**v, "num_matches": num_matches} for k, v in ML_model.items()}
                results[causal_method][num_matches] = get_causal_estimate(causal_data_dict, ML_model_dict, 
                                                                          causal_method, task, task_setup_dict, handle_imbalance, verbose_output)
                if print_results: print(f"{causal_method} (K={num_matches}) \t | \t ATE: {results[causal_method][num_matches]['ATE']:.4f}")

    if plot_results:

        num_subplots = len(causal_methods)
        if "matching" in causal_methods:
            num_subplots = num_subplots -1 + len(num_matches_list)
        num_subplot_cols = 3
        num_subplot_rows = int(np.ceil(num_subplots / num_subplot_cols))

        plt.figure(figsize=(20, 6))
        is_trimmed_str = "[Trimmed]" if is_trimmed else ""
        model_str = f"(Model: {list(ML_model.keys())[0]}{' - '+task_setup_dict.get('metric','median') if task=='survival' else ''}) " if ML_model else ""
        plt.suptitle(f"(Experiment {experiment_num}) Distribution of Estimated Treatment Effects with Different Causal Models {model_str}{is_trimmed_str}" +
                     f"| ({experiment_type})", 
                     fontsize=14)
        subplot_idx = 1
        for causal_method in causal_methods:
            
            if causal_method != "matching":
                # Plot the distribution of the estimated treatment effects
                plt.subplot(num_subplot_rows, num_subplot_cols, subplot_idx)
                plt.hist(results[causal_method]['ITE'], bins=100, alpha=0.5)
                plt.xlabel("Estimated Individual Treatment Effects", fontsize=8)
                plt.ylabel("Frequency", fontsize=8)
                plt.title(f"{causal_method} | (ATE: {results[causal_method]['ATE']:.3f})", fontsize=10)
                subplot_idx += 1
            else:
                for num_matches in num_matches_list:
                    # Plot the distribution of the estimated treatment effects
                    plt.subplot(num_subplot_rows, num_subplot_cols, subplot_idx)
                    plt.hist(results[causal_method][num_matches]['ITE'], bins=100, alpha=0.5)
                    plt.xlabel("Estimated Individual Treatment Effects", fontsize=8)
                    plt.ylabel("Frequency", fontsize=8)
                    plt.title(f"{causal_method} (K={num_matches}) | (ATE: {results[causal_method][num_matches]['ATE']:.3f})", fontsize=10)
                    subplot_idx += 1
        
        # Adjust vertical spacing between rows
        plt.subplots_adjust(hspace=1)  # Increase this value to add more space between rows
        plt.show()

    return results


def get_causal_estimate(causal_data_dict, model_type_dict, causal_method="matching", task="classification", task_setup_dict={}, handle_imbalance=True, verbose=True):
    if task not in ["classification", "survival"]:
        raise ValueError("Invalid task type")
    
    if causal_method == "t-learner":
        return train_t_learner(causal_data_dict, model_type_dict, task, task_setup_dict, handle_imbalance, verbose)
    elif causal_method == "s-learner":
        return train_s_learner(causal_data_dict, model_type_dict, task, task_setup_dict, handle_imbalance, verbose)
    elif causal_method == "x-learner":
        return train_x_learner(causal_data_dict, model_type_dict, task, task_setup_dict, handle_imbalance, verbose)
    elif causal_method == "matching":
        return train_matching_learner(causal_data_dict, model_type_dict, task, task_setup_dict, verbose)
    else:
        raise ValueError("Invalid causal method")


############################################################
################# Positivity Test ##########################
############################################################


def plot_positivity_check(causal_data_dict, experiment_num, experiment_type, verbose=True):

    causal_data = CausalModel(
        Y=causal_data_dict['Y']['total'][:, 1],     # Outcome
        D=causal_data_dict['A']['total'],           # Treatment
        X=causal_data_dict['X']['total']            # Covariates
    )


    # Estimate propensity scores
    causal_data.est_propensity()

    # Access propensity scores
    propensity_scores = causal_data.propensity['fitted']

    # Plot the distribution of propensity scores
    plt.hist(propensity_scores[causal_data_dict['A']['total'] == 1], bins=30, alpha=0.5, label="Treated")
    plt.hist(propensity_scores[causal_data_dict['A']['total'] == 0], bins=30, alpha=0.5, label="Control")
    plt.xlabel("Propensity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title(f"(Experiment {experiment_num}) Distribution of Propensity Scores | ({experiment_type})")
    plt.show()


    # Perform matching
    causal_data.est_via_matching(matches=10, bias_adj=False)

    # Perform OLS regression adjustment
    causal_data.est_via_ols()

    # Perform inverse probability weighting (IPW)
    causal_data.est_via_weighting()

    if verbose:
        # View the results
        print(causal_data.estimates)
