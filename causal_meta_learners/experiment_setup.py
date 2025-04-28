import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import pickle
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



class PatientData:
    def __init__(self, people_dict, data_df, experiment_type="Composite Event", 
                 task="classification", horizon=12, non_adherence_threshold=1./3, minimum_num_time_steps=4, low_occurrency_threshold=2, 
                 static_covariates_lst=['race_common_desc', 'ethnic_common_desc', 'legal_sex_common_desc', 'ed_lvl_common_desc'],
                 exclusion_static_covariates_lst=['ethnic_common_desc'],
                 continuous_covariates_lst=['age', 'predicted_PRO_MORTALITY_12MO', 'predicted_PRO_JAILSTAY_12MO', 'predicted_PRO_OVERDOSE_12MO',
                                            'predicted_PRO_302_12MO', 'predicted_PRO_SHELTER_STAY_12MO'],
                 post_hoc_covariates_lst=['covered_by_injectable', 'covered_by'],
                 random_seed=42, verbose=False):
        self.people_dict = people_dict
        self.data_df = data_df
        self.task = task
        self.experiment_type = experiment_type
        self.horizon = horizon
        self.non_adherence_threshold = non_adherence_threshold
        self.minimum_num_time_steps = minimum_num_time_steps
        self.low_occurrency_threshold = low_occurrency_threshold
        self.random_seed = random_seed
        self.verbose = verbose

        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # List of Static Covariates
        self.static_covariates_lst = static_covariates_lst
        self.exclusion_static_covariates_lst = exclusion_static_covariates_lst
        self.static_covariates_lst = [covariate for covariate in self.static_covariates_lst if covariate not in self.exclusion_static_covariates_lst]
        self._get_unique_static_covariate_combination()
        self._update_static_covariate_mapping('ed_lvl_common_desc')

        # List of Continuous Covariates
        self.continuous_covariates_lst = continuous_covariates_lst
        self.continuous_variable_snapshot_time = self.minimum_num_time_steps-2 # 0-indexed

        # List of Other Covariates for post-hoc analysis
        self.post_hoc_covariates_lst = post_hoc_covariates_lst

        # Get Data Setup
        self.static_covariates_matrix = self.get_static_covariates_matrix()
        self.first_event_time_type_lst = self.get_first_event_observed_time_and_event_indicator()
        self.covariate_matrix, self.non_adherence_matrix, self.outcome_matrix, self.post_hoc_covariates_matrix = self.preprocess_data_for_causal_inference()
        # self.age_column_idx = self.covariate_matrix.shape[1]-1
        self.continuous_column_indices = [self.covariate_names.index(cont_covariate) for cont_covariate in self.continuous_covariates_lst]


    def get_static_covariates_matrix(self):
        static_covariates_matrix = np.zeros((len(self.people_dict), len(self.static_covariates_lst)))

        for idx, person_key in enumerate(self.people_dict):
            person = self.people_dict[person_key]
            current_static_covariates = [self.static_covariate_mapping[covariate_name][person[covariate_name]] 
                                         for covariate_name in self.static_covariates_lst]
            static_covariates_matrix[idx, :] = np.array(current_static_covariates, dtype=int)
        
        return static_covariates_matrix
    

    def get_first_event_observed_time_and_event_indicator(self):
        # 302, jail, death
        first_event_lst = []

        for person in self.people_dict.values():
            # Any event: sum of the row >= 1 save the first index of this 
            # where the key is the time index and the value is the combined event
            person_event_indices = np.where(np.sum(person['combined_events'][:person['death_time']+1], axis=1) >= 1)[0]
            if len(person_event_indices) > 0:
                event_time_type = np.concatenate([[person_event_indices[0]], 
                                                person['combined_events'][person_event_indices[0]]])
            else:
                event_time_type = np.concatenate([[len(person['combined_events'][:person['death_time']+1])-1], 
                                                np.zeros(3)])
            first_event_lst.append(event_time_type)
            
        first_event_lst = np.array(first_event_lst)

        first_event_lst_simplified = np.zeros((first_event_lst.shape[0], 2))
        first_event_lst_simplified[:, 0] = first_event_lst[:, 0]
        self._update_lst_simplified(first_event_lst, first_event_lst_simplified) # update the first_event_lst_simplified in-place

        # ANY BAD EVENT
        first_event_lst_simplified[:, 1] = 1 * (first_event_lst_simplified[:, 1] > 0)

        # Need Horizon Adjustment for classification task (i.e. 0: no event, 1: event within horizon)
        # Note the task has to be corrected from left hand side too where we need to avoid data leakage.
        # i.e. all patients should have at least x timesteps and we are then defining the prediction task as
        # whether the event happens within the next horizon-x timesteps.
        # This method currently does not adjust for this.
        if self.task == "classification":
            first_event_lst_simplified[:, 1] = np.logical_and(first_event_lst_simplified[:, 0] <= self.horizon, first_event_lst_simplified[:, 1] != 0).astype(int)
        
        return first_event_lst_simplified


    def preprocess_data_for_causal_inference(self):

        # age_matrix = np.array([person['age'][int(event_time)] for person, event_time in zip(self.people_dict.values(), self.first_event_time_type_lst[:,0])])
        continous_covariate_matrix = np.array([[person[cont_covariate][self.continuous_variable_snapshot_time 
                                                                       if self.continuous_variable_snapshot_time < len(person[cont_covariate]) else -1]
                                                    for cont_covariate in self.continuous_covariates_lst]
                                                        for person in self.people_dict.values()])
        

        # Keep the list of other covariates for post-hoc analysis (from time 0 to snapshot time)
        post_hoc_covariates_matrix = {other_covariate: [person[other_covariate][:self.minimum_num_time_steps-1 
                                                                        if (self.minimum_num_time_steps-1) < len(person[other_covariate]) else len(person[other_covariate])] 
                                                            for person in self.people_dict.values()] 
                                                    for other_covariate in self.post_hoc_covariates_lst}
        

        # non-adherence matrix (treatment matrix)
        # convert non-adherence values to 0 and 1
        # More than 10 days of non-adherence in a month is considered non-adherent (Based on Median in previous analysis)
        non_adherence_matrix = [person['non_covered_days'][:int(event_time)+1] for person, event_time in zip(self.people_dict.values(), self.first_event_time_type_lst[:,0])]
        non_adherence_matrix = [np.array(list(map(lambda A_t: 1 if A_t > self.non_adherence_threshold else 0, non_adherence_patient))) for non_adherence_patient in non_adherence_matrix]


        # For X, Abar, Y matrices drop the low occuring static covariate combination indices
        # Get the low occuring covariate combination indices
        low_occurrency_static_covariate_combination_indices = self._get_low_occuring_indices(self.static_covariates_matrix, self.low_occurrency_threshold)
        if self.verbose:
            print(f"Number of people with less than {self.low_occurrency_threshold} covariate combination occurrences: {len(low_occurrency_static_covariate_combination_indices)}")
        # X = np.delete(X, low_occurrency_static_covariate_combination_indices, axis=0)
        static_covariates_matrix_updated = np.delete(self.static_covariates_matrix, low_occurrency_static_covariate_combination_indices, axis=0)
        continous_covariate_matrix = np.delete(continous_covariate_matrix, low_occurrency_static_covariate_combination_indices, axis=0)
        post_hoc_covariates_matrix = {other_covariate: [other_value for idx, other_value in enumerate(other_value_lst) 
                                                            if idx not in low_occurrency_static_covariate_combination_indices]
                                                for other_covariate, other_value_lst in post_hoc_covariates_matrix.items()}
        A_bar = [A_bar_i for index, A_bar_i in enumerate(non_adherence_matrix) if index not in low_occurrency_static_covariate_combination_indices]
        Y = np.delete(self.first_event_time_type_lst, low_occurrency_static_covariate_combination_indices, axis=0)


        # Get indices of people with less than minimum_num_time_steps
        low_num_time_steps_indices = np.where(np.array([len(A_bar_i) for A_bar_i in A_bar]) < self.minimum_num_time_steps)[0]

        # Print Information of the data after preprocessing
        ratio_of_people_with_events_and_minimum_num_timesteps = sum([1 for A_bar_i, event_type in zip(A_bar, Y[:,1]) 
                                                                if len(A_bar_i) >= self.minimum_num_time_steps and event_type!=0]) \
                                                                    / len(Y[Y[:,1]!=0])
        ratio_of_people_with_minimum_num_timesteps = sum([1 for A_bar_i in A_bar 
                                                    if len(A_bar_i) >= self.minimum_num_time_steps]) \
                                                        / len(A_bar)
        if self.verbose:
            print(f"Number of people with less than {self.minimum_num_time_steps} timesteps: {len(low_num_time_steps_indices)}")
            print(f"Percentage of people with an event happening and at least {self.minimum_num_time_steps} timesteps: {ratio_of_people_with_events_and_minimum_num_timesteps*100:.2f}%")
            print(f"Percentage of people with at least {self.minimum_num_time_steps} timesteps: {ratio_of_people_with_minimum_num_timesteps*100:.2f}%")

        # X = np.delete(X, low_num_time_steps_indices, axis=0)
        static_covariates_matrix_updated = np.delete(static_covariates_matrix_updated, low_num_time_steps_indices, axis=0)
        continous_covariate_matrix       = np.delete(continous_covariate_matrix,       low_num_time_steps_indices, axis=0)
        post_hoc_covariates_matrix = {other_covariate: [other_value for idx, other_value in enumerate(other_value_lst)
                                                            if idx not in low_num_time_steps_indices]
                                                for other_covariate, other_value_lst in post_hoc_covariates_matrix.items()}
        A_bar = [A_bar_i for index, A_bar_i in enumerate(A_bar) if index not in low_num_time_steps_indices]
        Y = np.delete(Y, low_num_time_steps_indices, axis=0)

        if self.verbose:
            print(f"Number of people after removing low occurring static covariate combination and low number of timesteps: {Y.shape[0]}")
            print(f"Number of people removed: {len(low_occurrency_static_covariate_combination_indices) + len(low_num_time_steps_indices)} / {len(self.people_dict)} = " + 
                f"{(len(low_occurrency_static_covariate_combination_indices) + len(low_num_time_steps_indices))/len(self.people_dict)*100:.2f}%")
        

        # Change outcome 4->2, 5->3, 6->3
        if self.verbose: print(f"Number of people with event type adjustment: {len(Y[Y[:,1]>3])}")
        Y[Y[:,1] == 4, 1] = 2
        Y[Y[:,1] == 5, 1] = 3
        Y[Y[:,1] == 6, 1] = 3

        # Initialize OneHotEncoder with drop='first' to avoid collinearity
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        static_covariates_matrix_one_hot = encoder.fit_transform(static_covariates_matrix_updated)

        # Generate one-hot encoded feature names
        one_hot_feature_names = encoder.get_feature_names_out(self.static_covariates_lst)

        # combined covariates
        X = np.concatenate([static_covariates_matrix_one_hot, continous_covariate_matrix], axis=1)

        self.covariate_names = list(one_hot_feature_names) + self.continuous_covariates_lst

        return X, A_bar, Y, post_hoc_covariates_matrix


    def get_causal_data_setup_for_each_experiment(self, experiment_num=0, test_size=0.2, random_state=42):

        if experiment_num == 1:
            # Get f(A_bar) = mean(A_bar[:-1])
            mean_past_adherence = np.array([np.mean(A_bar_i[:self.horizon-1] if len(A_bar_i)>self.horizon else A_bar_i[:-1]) for A_bar_i in self.non_adherence_matrix])

            covariate_matrix_new = np.concatenate([self.covariate_matrix, mean_past_adherence.reshape(-1, 1)], axis=1)
            non_adherence_treatment = np.array([A_bar_i[self.horizon] if len(A_bar_i)>self.horizon else A_bar_i[-1] for A_bar_i in self.non_adherence_matrix])

            self.covariate_names_for_experiment_full = self.covariate_names + ['mean_past_nonadherence']
        
        elif experiment_num == 2:
            # Get f(A_bar) = mean(A_bar[-4:-1])
            mean_past_adherence = np.array([np.mean(A_bar_i[self.horizon-4:self.horizon-1] if len(A_bar_i)>self.horizon else A_bar_i[-4:-1]) for A_bar_i in self.non_adherence_matrix])

            covariate_matrix_new = np.concatenate([self.covariate_matrix, mean_past_adherence.reshape(-1, 1)], axis=1)
            non_adherence_treatment = np.array([A_bar_i[self.horizon-1] if len(A_bar_i)>self.horizon else A_bar_i[-1] for A_bar_i in self.non_adherence_matrix])

            self.covariate_names_for_experiment_full = self.covariate_names + ['mean_past_3_nonadherence']

        elif experiment_num == 3:
            # Get f(A_bar) = A_bar[-4:-1]
            past_adherence_squence = np.array([np.array(A_bar_i[self.horizon-4:self.horizon-1] if len(A_bar_i)>self.horizon else A_bar_i[-4:-1]) for A_bar_i in self.non_adherence_matrix])

            covariate_matrix_new = np.concatenate([self.covariate_matrix, past_adherence_squence], axis=1)
            non_adherence_treatment = np.array([A_bar_i[self.horizon-1] if len(A_bar_i)>self.horizon else A_bar_i[-1] for A_bar_i in self.non_adherence_matrix])

            self.covariate_names_for_experiment_full = self.covariate_names + [f'past_{i}_nonadherence' for i in range(3, 0, -1)]

        elif experiment_num == 4:
            # Get f(A_bar) = A_bar[:3]
            past_adherence_squence = np.array([np.array(A_bar_i[:self.continuous_variable_snapshot_time]) for A_bar_i in self.non_adherence_matrix])

            covariate_matrix_new = np.concatenate([self.covariate_matrix, past_adherence_squence], axis=1)
            non_adherence_treatment = np.array([A_bar_i[self.continuous_variable_snapshot_time] for A_bar_i in self.non_adherence_matrix])

            self.covariate_names_for_experiment_full = self.covariate_names + [f'past_nonadherence_at_{i}' for i in range(2)]
        
        elif experiment_num == "SA":
            past_adherence_squence = np.array([np.array(A_bar_i[:self.continuous_variable_snapshot_time]) for A_bar_i in self.non_adherence_matrix])

            covariate_matrix_new = np.concatenate([self.covariate_matrix, past_adherence_squence], axis=1)
            non_adherence_treatment = np.array([A_bar_i[self.continuous_variable_snapshot_time] for A_bar_i in self.non_adherence_matrix])

            self.covariate_names_for_experiment_full = self.covariate_names + [f'past_nonadherence_at_{i}' for i in range(2)]

        else:
            raise ValueError("Invalid experiment number")
        
        self.covariate_names_for_experiment_current = self.covariate_names_for_experiment_full.copy()
        
        if experiment_num == "SA" and self.task != "survival":
            self.task = "survival"
            print("Task is changed to survival analysis")
        elif experiment_num != "SA" and self.task == "survival":
            self.task = "classification"
            print("Task is changed to classification")
        
        if self.verbose:
            print(f"Experiment {experiment_num}:")
            print(f"Covariate: {covariate_matrix_new.shape}, Non-Adherence: {len(non_adherence_treatment)}, Outcome: {self.outcome_matrix.shape}")

        return self._get_training_test_setup(covariate_matrix_new, non_adherence_treatment, self.outcome_matrix, self.post_hoc_covariates_matrix,
                                             self.continuous_column_indices, experiment_num, test_size=test_size, random_state=random_state)


    def _get_training_test_setup(self, covariate_matrix, non_adherence_treatment, outcome_matrix, post_hoc_covariates_matrix, continuous_column_indices, 
                                 experiment_num=1, test_size=0.2, random_state=42):
        

        post_hoc_values = list(post_hoc_covariates_matrix.values())
        
        split_data = train_test_split(covariate_matrix, non_adherence_treatment, outcome_matrix, *post_hoc_values,
                                      test_size=test_size, random_state=random_state)
        
        # Unpack the first six outputs (X_train, X_test, A_train, A_test, Y_train, Y_test)
        X_train, X_test, A_train, A_test, Y_train, Y_test = split_data[:6]
        # Handle the post_hoc splits
        post_hoc_split = split_data[6:]
        
        # Reconstruct the post_hoc_train and post_hoc_test dictionaries
        post_hoc_train = {key: post_hoc_split[i * 2] for i, key in enumerate(post_hoc_covariates_matrix.keys())}
        post_hoc_test  = {key: post_hoc_split[i * 2 + 1] for i, key in enumerate(post_hoc_covariates_matrix.keys())}
        
        # normalize the continuous covariate(s) in the training and testing set
        scaler = StandardScaler()
        X_train[:, continuous_column_indices] = scaler.fit_transform(X_train[:, continuous_column_indices])
        X_test[:, continuous_column_indices] = scaler.transform(X_test[:, continuous_column_indices])


        if experiment_num == "SA":
            # Adjust time index of the outcome matrix for survival analysis
            Y_train[:, 0] = Y_train[:, 0] + 1 - 3
            Y_test[:, 0]  = Y_test[:, 0]  + 1 - 3

        X_total = np.concatenate([X_train, X_test], axis=0) # for the purpose of the causal inference
        A_total = np.concatenate([A_train, A_test], axis=0) # for the purpose of the causal inference
        Y_total = np.concatenate([Y_train, Y_test], axis=0) # for the purpose of the causal inference
        post_hoc_total = {key: post_hoc_train[key] + post_hoc_test[key] for key in post_hoc_train.keys()}


        # rearrange data for no columns having standard deviation of 0 (per subset of A={0, 1}) in the training set
        covariate_column_names = self.covariate_names_for_experiment_full
        if experiment_num == "SA":
            if X_total[A_total == 0].std(axis=0).min() == 0 or X_total[A_total == 1].std(axis=0).min() == 0:
                print(f"[Random-Seed:{self.random_seed}] Standard deviation of columns in the total set is 0 (for one of the treatment assignments). Rearranging the data...")
                split_row_idx = X_train.shape[0]
                filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc, new_split_row_idx = self._trim_data_for_positivity(X_total,
                                                                                                                                               A_total,
                                                                                                                                               Y_total,
                                                                                                                                               post_hoc_total,
                                                                                                                                               self.continuous_column_indices,
                                                                                                                                               split_row_idx=split_row_idx,
                                                                                                                                               experiment_num=experiment_num,
                                                                                                                                               exclude_continuous=True,
                                                                                                                                               verbose_output=False)
                X_train, X_test = filtered_X_adjusted[:new_split_row_idx], filtered_X_adjusted[new_split_row_idx:]
                A_train, A_test = filtered_A[:new_split_row_idx], filtered_A[new_split_row_idx:]
                Y_train, Y_test = filtered_Y[:new_split_row_idx], filtered_Y[new_split_row_idx:]
                post_hoc_train = {key: filtered_post_hoc[key][:new_split_row_idx] for key in filtered_post_hoc.keys()}
                post_hoc_test = {key: filtered_post_hoc[key][new_split_row_idx:] for key in filtered_post_hoc.keys()}
                X_total = np.concatenate([X_train, X_test], axis=0) # for the purpose of the causal inference
                A_total = np.concatenate([A_train, A_test], axis=0) # for the purpose of the causal inference
                Y_total = np.concatenate([Y_train, Y_test], axis=0) # for the purpose of the causal inference
                post_hoc_total = {key: post_hoc_train[key] + post_hoc_test[key] for key in post_hoc_train.keys()}
                covariate_column_names = self.covariate_names_for_experiment_trimmed.copy()

            elif X_train[A_train == 0].std(axis=0).min() == 0 or X_train[A_train == 1].std(axis=0).min() == 0:
                print(f"[Random-Seed:{self.random_seed}] Standard deviation of columns in the training set is 0 (for one of the treatment assignments). Rearranging the data...")
                split_row_idx = X_train.shape[0]
                X_total, _, A_total, Y_total, post_hoc_total = self._ensure_non_zero_variance_in_subsets(X_total, X_total.copy(),
                                                                                                         A_total, Y_total, post_hoc_total,
                                                                                                         split_row_idx)
                X_train, X_test = X_total[:split_row_idx], X_total[split_row_idx:]
                A_train, A_test = A_total[:split_row_idx], A_total[split_row_idx:]
                Y_train, Y_test = Y_total[:split_row_idx], Y_total[split_row_idx:]
                post_hoc_train = {key: post_hoc_total[key][:split_row_idx] for key in post_hoc_total.keys()}
                post_hoc_test = {key: post_hoc_total[key][split_row_idx:] for key in post_hoc_total.keys()}
        
        if self.verbose:
            print(f"Training set: X: {X_train.shape}, A: {len(A_train)}, Y: {Y_train.shape}")
            print(f"Testing set: X: {X_test.shape}, A: {len(A_test)}, Y: {Y_test.shape}")
            print(f"Total set: X: {X_total.shape}, A: {len(A_total)}, Y: {Y_total.shape}")

        return {"X": {'train': X_train, 'test': X_test, 'total': X_total},
                "A": {'train': A_train, 'test': A_test, 'total': A_total},
                "Y": {'train': Y_train, 'test': Y_test, 'total': Y_total},
                "post_hoc": {'train': post_hoc_train, 'test': post_hoc_test, 'total': post_hoc_total},
                "covariate_names": covariate_column_names}


    def adjust_data_for_positivity(self, causal_data_dict, experiment_num, exclude_continuous=False, verbose_output=True):
        split_row_idx = causal_data_dict['X']['train'].shape[0]
        filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc, new_split_row_idx = self._trim_data_for_positivity(causal_data_dict['X']['total'],
                                                                                                                                       causal_data_dict['A']['total'],
                                                                                                                                       causal_data_dict['Y']['total'],
                                                                                                                                       causal_data_dict['post_hoc']['total'],
                                                                                                                                       self.continuous_column_indices,
                                                                                                                                       split_row_idx=split_row_idx,
                                                                                                                                       experiment_num=experiment_num,
                                                                                                                                       exclude_continuous=exclude_continuous,
                                                                                                                                       verbose_output=verbose_output)
        
        return {"X": {'total': filtered_X_adjusted, 'total_unadjusted': filtered_X, 
                      'train': filtered_X_adjusted[:new_split_row_idx], 'test': filtered_X_adjusted[new_split_row_idx:]},
                "A": {'total': filtered_A, 'train': filtered_A[:new_split_row_idx], 'test': filtered_A[new_split_row_idx:]},
                "Y": {'total': filtered_Y, 'train': filtered_Y[:new_split_row_idx], 'test': filtered_Y[new_split_row_idx:]}, 
                "post_hoc": {"train": {key: filtered_post_hoc[key][:new_split_row_idx] for key in filtered_post_hoc.keys()},
                             "test":  {key: filtered_post_hoc[key][new_split_row_idx:] for key in filtered_post_hoc.keys()},
                             "total": {key: filtered_post_hoc[key] for key in filtered_post_hoc.keys()}}
                }
    

    def _trim_data_for_positivity(self, X_total, A_total, Y_total, post_hoc_total, continuous_column_indices, 
                                  split_row_idx=1, experiment_num=0, exclude_continuous=False, verbose_output=True):
        # Convert X_total to a DataFrame for easier grouping
        X_df = pd.DataFrame(X_total)
        X_df['A'] = A_total  # Add treatment indicator to the DataFrame

        # Handle continuous columns
        if exclude_continuous:
            continuous_column_values = X_df[continuous_column_indices].values
            X_df[continuous_column_indices] = 0  # Temporarily ignore continuous columns for grouping

        # Group by unique rows in X_total and analyze their A values
        grouped = X_df.groupby(list(range(X_total.shape[1])))['A']
        always_treated_combinations = grouped.nunique()[grouped.nunique() == 1][grouped.mean() == 1].index
        always_control_combinations = grouped.nunique()[grouped.nunique() == 1][grouped.mean() == 0].index

        # Convert these combinations back to numpy arrays
        always_treated_rows = np.array([list(comb) for comb in always_treated_combinations])
        always_control_rows = np.array([list(comb) for comb in always_control_combinations])

        # Print statistics
        if verbose_output:
            print(f"Number of always treated row combinations: {len(always_treated_rows)}")
            print(f"Number of always control row combinations: {len(always_control_rows)}")

        # Filter rows
        is_always_treated = X_df.iloc[:, :-1].apply(tuple, axis=1).isin(always_treated_combinations)
        is_always_control = X_df.iloc[:, :-1].apply(tuple, axis=1).isin(always_control_combinations)
        rows_to_remove = is_always_treated | is_always_control

        if exclude_continuous:
            X_df[continuous_column_indices] = continuous_column_values  # Restore contunious columns values

        # Get original indices of rows to keep
        rows_to_keep = ~rows_to_remove
        keep_indices = np.arange(len(X_total))[rows_to_keep]

        # Calculate the new split row index
        new_split_row_idx = np.searchsorted(keep_indices, split_row_idx)

        # Filtered dataset
        filtered_X_df = X_df[rows_to_keep]
        filtered_X = filtered_X_df.iloc[:, :-1].values
        filtered_A = filtered_X_df['A'].values
        filtered_Y = Y_total[rows_to_keep]

        # Filter post_hoc
        filtered_post_hoc = {
            key: [values[idx] for idx in np.where(rows_to_keep)[0]]
            for key, values in post_hoc_total.items()
        }

        # Drop constant columns
        constant_columns = np.where(filtered_X.std(axis=0) == 0)[0]
        filtered_X_adjusted = np.delete(filtered_X, constant_columns, axis=1)

        # Ensure non-zero variance in each column for two subsets of filtered_X_adjusted defined by filtered_A
        filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc = self._ensure_non_zero_variance_in_subsets(filtered_X, 
                                                                                                                               filtered_X_adjusted, 
                                                                                                                               filtered_A, 
                                                                                                                               filtered_Y, 
                                                                                                                               filtered_post_hoc, 
                                                                                                                               new_split_row_idx)

        self.covariate_names_for_experiment_trimmed = self.covariate_names_for_experiment_full.copy()
        # Remove constant columns from the list of covariate names
        for idx in sorted(constant_columns, reverse=True):
            self.covariate_names_for_experiment_trimmed.pop(idx)
        self.covariate_names_for_experiment_current = self.covariate_names_for_experiment_trimmed.copy()

        # Validate row counts
        assert X_total.shape[0] == is_always_treated.sum() + is_always_control.sum() + filtered_X.shape[0], "Row counts do not add up!"

        # Verbose output
        if verbose_output:
            print(f"Total rows: {X_total.shape[0]}")
            print(f"Rows always treated: {is_always_treated.sum()}")
            print(f"Rows always control: {is_always_control.sum()}")
            print(f"Filtered rows: {filtered_X.shape[0]}")
            print(f"Number of columns before filtering: {X_total.shape[1]}")
            print(f"Number of columns after filtering: {filtered_X_adjusted.shape[1]}")
            print(f"Original split row index: {split_row_idx}")
            print(f"New split row index: {new_split_row_idx}")

            # Plot histogram
            plt.figure(figsize=(10, 6))
            plt.hist(A_total[is_always_treated], bins=2, alpha=0.7, label='Always Treated')
            plt.hist(A_total[is_always_control], bins=2, alpha=0.7, label='Always Control')
            plt.title(f"Experiment {experiment_num}: Always Treated vs Control | {self.experiment_type}")
            plt.xlabel("A")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()

        return filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc, new_split_row_idx


    def _ensure_non_zero_variance_in_subsets(self, filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc, new_split_row_idx):
        """Ensures non-zero variance in each column for two subsets of filtered_X_adjusted defined by filtered_A.

        Args:
            filtered_X: The original feature matrix.
            filtered_X_adjusted: The filtered and adjusted feature matrix.
            filtered_A: The binary array indicating the subset membership.
            filtered_Y: The target variable array.
            filtered_post_hoc: Dictionary of post_hoc covariates.
            new_split_row_idx: The index separating the training and test sets.

        Returns:
            The modified filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, and filtered_post_hoc.
        """
        for col_idx in range(filtered_X_adjusted.shape[1]):
            # Subset 0
            subset_0_train_idx = np.where((filtered_A[:new_split_row_idx] == 0))[0]
            subset_0_test_idx = np.where((filtered_A[new_split_row_idx:] == 0))[0] + new_split_row_idx

            while np.var(filtered_X_adjusted[subset_0_train_idx, col_idx]) < 1e-4:
                # Swap the last train row with the first test row
                filtered_X_adjusted[[subset_0_train_idx[-1], subset_0_test_idx[0]]] = filtered_X_adjusted[[subset_0_test_idx[0], subset_0_train_idx[-1]]]
                filtered_X[[subset_0_train_idx[-1], subset_0_test_idx[0]]] = filtered_X[[subset_0_test_idx[0], subset_0_train_idx[-1]]]
                filtered_A[[subset_0_train_idx[-1], subset_0_test_idx[0]]] = filtered_A[[subset_0_test_idx[0], subset_0_train_idx[-1]]]
                filtered_Y[[subset_0_train_idx[-1], subset_0_test_idx[0]]] = filtered_Y[[subset_0_test_idx[0], subset_0_train_idx[-1]]]

                for key in filtered_post_hoc.keys():
                    filtered_post_hoc[key][subset_0_train_idx[-1]], filtered_post_hoc[key][subset_0_test_idx[0]] = \
                        filtered_post_hoc[key][subset_0_test_idx[0]], filtered_post_hoc[key][subset_0_train_idx[-1]]

                # Update the test set index, but keep the original train set index for the next variance check
                subset_0_test_idx = subset_0_test_idx[1:]

            # Subset 1
            subset_1_train_idx = np.where((filtered_A[:new_split_row_idx] == 1))[0]
            subset_1_test_idx = np.where((filtered_A[new_split_row_idx:] == 1))[0] + new_split_row_idx

            while np.var(filtered_X_adjusted[subset_1_train_idx, col_idx]) < 1e-4:
                # Swap the last train row with the first test row
                filtered_X_adjusted[[subset_1_train_idx[-1], subset_1_test_idx[0]]] = filtered_X_adjusted[[subset_1_test_idx[0], subset_1_train_idx[-1]]]
                filtered_X[[subset_1_train_idx[-1], subset_1_test_idx[0]]] = filtered_X[[subset_1_test_idx[0], subset_1_train_idx[-1]]]
                filtered_A[[subset_1_train_idx[-1], subset_1_test_idx[0]]] = filtered_A[[subset_1_test_idx[0], subset_1_train_idx[-1]]]
                filtered_Y[[subset_1_train_idx[-1], subset_1_test_idx[0]]] = filtered_Y[[subset_1_test_idx[0], subset_1_train_idx[-1]]]

                for key in filtered_post_hoc.keys():
                    filtered_post_hoc[key][subset_1_train_idx[-1]], filtered_post_hoc[key][subset_1_test_idx[0]] = \
                        filtered_post_hoc[key][subset_1_test_idx[0]], filtered_post_hoc[key][subset_1_train_idx[-1]]

                # Update the test set index, but keep the original train set index for the next variance check
                subset_1_test_idx = subset_1_test_idx[1:]

        return filtered_X, filtered_X_adjusted, filtered_A, filtered_Y, filtered_post_hoc


    def _update_lst_simplified(self, first_event_lst, first_event_lst_simplified):
        # Simplify list of first events 
        # (0: no event, 1: 302, 2: jail, 3: death, 4: 302+jail, 5: 302+death, 6: jail+death, 7: 302+jail+death)
        # first element is timestep of first event, and second element is the type of event
        for idx in range(first_event_lst.shape[0]):
            if sum(first_event_lst[idx, 1:]) == 0:
                first_event_lst_simplified[idx, 1] = 0
            elif sum(first_event_lst[idx, 1:]) == 1:
                if first_event_lst[idx, 1] == 1:
                    first_event_lst_simplified[idx, 1] = 1 #302
                elif first_event_lst[idx, 2] == 1:
                    first_event_lst_simplified[idx, 1] = 2 #jail
                elif first_event_lst[idx, 3] == 1:
                    first_event_lst_simplified[idx, 1] = 3 #death
            elif sum(first_event_lst[idx, 1:]) == 2:
                if first_event_lst[idx, 1] == 1 and first_event_lst[idx, 2] == 1: #302, jail
                    first_event_lst_simplified[idx, 1] = 4
                elif first_event_lst[idx, 1] == 1 and first_event_lst[idx, 3] == 1: #302, death
                    first_event_lst_simplified[idx, 1] = 5
                elif first_event_lst[idx, 2] == 1 and first_event_lst[idx, 3] == 1: #jail, death
                    first_event_lst_simplified[idx, 1] = 6
            else: #if sum(first_event_lst[idx, 1:]) == 3: #302, jail, death
                first_event_lst_simplified[idx, 1] = 7


    def _get_unique_static_covariate_combination(self):
        self.unique_static_covariate_values = {covariate_name: set(self.data_df[covariate_name]) for covariate_name in self.static_covariates_lst}

        # create a mapping from the unique values to an integer (and vice versa)
        self.static_covariate_mapping = {covariate_name: {value: idx for idx, value in enumerate(self.unique_static_covariate_values[covariate_name])} 
                                         for covariate_name in self.static_covariates_lst}
        self.static_covariate_mapping_reverse = {covariate_name: {idx: value for idx, value in enumerate(self.unique_static_covariate_values[covariate_name])} 
                                                 for covariate_name in self.static_covariates_lst}
        
    
    def _update_static_covariate_mapping(self, covariate_name):
        """
        Updates a static_covariate_mapping dictionary for a given covariate_name,
        preserving distinct values and reordering sequentially.

        Args:
            static_covariate_mapping (dict): The dictionary containing covariate mappings.
            covariate_name (str): The specific covariate name to update.

        Returns:
            dict: The updated static_covariate_mapping.
        """
        # Extract the covariate mapping for the given covariate_name
        covariate_mapping = self.static_covariate_mapping[covariate_name]

        # Extract original values
        original_values = list(covariate_mapping.values())

        # Sort unique values, but maintain distinct floating-point values
        unique_values = sorted(set(original_values))

        # Identify all distinct floating-point values to preserve
        distinct_values = {val for val in original_values if isinstance(val, float)}

        # Reorder the mapping, skipping distinct values
        value_mapping = {}
        new_index = 0
        for val in unique_values:
            if val in distinct_values:
                value_mapping[val] = val  # Preserve distinct floating-point values
            else:
                value_mapping[val] = new_index
                new_index += 1

        # Apply the mapping to update the covariate
        self.static_covariate_mapping[covariate_name] = {
            key: value_mapping[value] for key, value in covariate_mapping.items()
        }

        # Update the reverse mapping
        self.static_covariate_mapping_reverse = {covariate_name_: {value: key for key, value in self.static_covariate_mapping[covariate_name_].items()} 
                                                 for covariate_name_ in self.static_covariates_lst}


    def _get_low_occuring_indices(self, static_covariates_matrix, low_occurrency_threshold=None):

        if low_occurrency_threshold is None:
            low_occurrency_threshold = self.low_occurrency_threshold
        elif low_occurrency_threshold != self.low_occurrency_threshold:
            print(f"Using low_occurrency_threshold: {low_occurrency_threshold} - Overriding the class variable ({self.low_occurrency_threshold})")

        unique_static_covariate_combination = np.unique(static_covariates_matrix, axis=0, return_counts=True)

        # get the indices of the people that have the static covariate combinatio
        count_of_each_static_covariate_combination = [
            unique_static_covariate_combination[1][np.where(np.all(unique_static_covariate_combination[0] == person_covariate, axis=1))[0][0]]
                if np.any(np.all(unique_static_covariate_combination[0] == person_covariate, axis=1)) else 0 # else should not happen
                for person_covariate in static_covariates_matrix
        ]

        assert min(count_of_each_static_covariate_combination) > 0

        # get the indices of the people that have the static covariate combination that appears less than low_occurrency_threshold times
        low_occurrency_static_covariate_combination_indices = np.where(np.array(count_of_each_static_covariate_combination) < low_occurrency_threshold)[0]

        return low_occurrency_static_covariate_combination_indices
    

    

