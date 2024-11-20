import editdistance
from nltk.translate.bleu_score import sentence_bleu
from scipy.spatial import distance
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np
import torch
import csv
from captum.attr import IntegratedGradients, Saliency
import matplotlib.pyplot as plt
import os
import pandas as pd

from collections import Counter
from math import log2
from scipy.stats import ks_2samp
import sacrebleu



smoothie = SmoothingFunction().method1
device = torch.device("cpu")


def create_od_set(test_trajs):
    test_od_dict = {}
    for i in range(len(test_trajs)):
        if (test_trajs[i][0], test_trajs[i][-1]) in test_od_dict.keys():
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])].append(i)
        else:
            test_od_dict[(test_trajs[i][0], test_trajs[i][-1])] = [i]
    return test_od_dict


def evaluate_edit_dist(test_trajs, learner_trajs, test_od_dict):
    edit_dist_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            min_edit_dist = 1.0
            for test in test_od_trajs:
                edit_dist = editdistance.eval(test, learner) / len(test)
                min_edit_dist = edit_dist if edit_dist < min_edit_dist else min_edit_dist
            edit_dist_list.append(min_edit_dist)
    return np.mean(edit_dist_list)


# def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
#     bleu_score_list = []
#     for od in test_od_dict.keys():
#         idx_list = test_od_dict[od]
#         # get unique reference
#         test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
#         test_od_trajs = [traj.split('_') for traj in test_od_trajs]
#         learner_od_trajs = [learner_trajs[i] for i in idx_list]
#         for learner in learner_od_trajs:
#             # print(test_od_trajs)
#             # print(learner)
#             bleu_score = sentence_bleu(test_od_trajs, learner, smoothing_function=smoothie)
#             bleu_score_list.append(bleu_score)
#     return np.mean(bleu_score_list)
def evaluate_bleu_score(test_trajs, learner_trajs, test_od_dict):
    bleu_score_list = []
    for od in test_od_dict.keys():
        idx_list = test_od_dict[od]
        # Get unique reference trajectories
        test_od_trajs = set(['_'.join(test_trajs[i]) for i in idx_list])
        test_od_trajs = [traj.split('_') for traj in test_od_trajs]
        learner_od_trajs = [learner_trajs[i] for i in idx_list]
        for learner in learner_od_trajs:
            # SacréBLEU expects references as list of strings and hypothesis as a single string
            references = [' '.join(ref) for ref in test_od_trajs]
            hypothesis = ' '.join(learner)
            bleu = sacrebleu.corpus_bleu([hypothesis], [references]).score
            bleu_score_list.append(bleu)
    return np.mean(bleu_score_list)


def evaluate_dataset_dist(test_trajs, learner_trajs):
    test_trajs_str = ['_'.join(traj) for traj in test_trajs]
    # print('test trajs str len', len(test_trajs_str))
    test_trajs_set = set(test_trajs_str)
    # print('test trajs set len', len(test_trajs_set))
    test_trajs_dict = dict(zip(list(test_trajs_set), range(len(test_trajs_set))))
    test_trajs_label = [test_trajs_dict[traj] for traj in test_trajs_str]
    test_trajs_label.append(0)
    test_p = np.histogram(test_trajs_label)[0] / len(test_trajs_label)

    pad_idx = len(test_trajs_set)
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]
    learner_trajs_label = [test_trajs_dict.get(traj, pad_idx) for traj in learner_trajs_str]
    learner_p = np.histogram(learner_trajs_label)[0] / len(learner_trajs_label)
    return distance.jensenshannon(test_p, learner_p)


def evaluate_log_prob(test_traj, model):
    log_prob_list = []
    for episode in test_traj:
        des = torch.LongTensor([episode[-1].next_state]).long().to(device)
        log_prob = 0
        for x in episode:
            with torch.no_grad():
                next_prob = torch.log(model.get_action_prob(torch.LongTensor([x.cur_state]).to(device), des)).squeeze()
            next_prob_np = next_prob.detach().cpu().numpy()
            log_prob += next_prob_np[x.action]
        log_prob_list.append(log_prob)
    print(np.mean(log_prob_list))
    return np.mean(log_prob_list)



def evaluate_length_distribution_similarity(expert_trajs, learner_trajs):
    # Compute lengths
    expert_lengths = [len(traj) for traj in expert_trajs]
    learner_lengths = [len(traj) for traj in learner_trajs]
    
    # Compute summary statistics
    expert_mean = np.mean(expert_lengths)
    learner_mean = np.mean(learner_lengths)
    expert_std = np.std(expert_lengths)
    learner_std = np.std(learner_lengths)
    
    # Perform KS test
    ks_statistic, ks_pvalue = ks_2samp(expert_lengths, learner_lengths)
    
    results = {
        'Expert Mean Length': expert_mean,
        'Learner Mean Length': learner_mean,
        'Expert Length Std': expert_std,
        'Learner Length Std': learner_std,
        'KS Statistic': ks_statistic,
        'KS P-Value': ks_pvalue
    }
    
    return expert_mean,learner_mean,expert_std,learner_std,ks_statistic,ks_pvalue



def evaluate_train_edit_dist(train_traj, learner_traj):
    """This function is used to keep the training epoch with the best edit distance performance on the training data"""
    test_od_dict = create_od_set(train_traj)
    edit_dist = evaluate_edit_dist(train_traj, learner_traj, test_od_dict)
    return edit_dist


def evaluate_metrics(test_traj, learner_traj,policy):
    test_od_dict = create_od_set(test_traj)
    edit_dist = evaluate_edit_dist(test_traj, learner_traj, test_od_dict)
    bleu_score = evaluate_bleu_score(test_traj, learner_traj, test_od_dict)
    js_dist = evaluate_dataset_dist(test_traj, learner_traj)
    # avg_log_likeli = evaluate_log_prob(test_traj, policy)
    expert_mean,learner_mean,expert_std,learner_std,ks_statistic,ks_pvalue = evaluate_length_distribution_similarity(test_traj, learner_traj)

    # print('edit dist', edit_dist)
    # print('bleu score', bleu_score)
    # print('js distance', js_dist)
    # print('average_log_likelihood', avg_log_likeli)
    # print('perplexity', perplexity)
    # print(f'expert_mean {expert_mean}, learner_mean {learner_mean}')
    # print(f'expert_std {expert_std},learner_std {learner_std}')
    return edit_dist, bleu_score, js_dist,expert_mean,learner_mean,expert_std,learner_std



def aggregate_attributions(attributions_all):
    total_attr = None
    count = 0

    for attr_traj in attributions_all:
        for attr_step in attr_traj:
            if total_attr is None:
                total_attr = attr_step.copy()
            else:
                total_attr += attr_step
            count += 1

    avg_attr = total_attr / count if count > 0 else total_attr
    return avg_attr

def visualize_attribution(attr, title):
    attr_sum = np.sum(attr, axis=0)  # Sum over channels
    plt.figure(figsize=(6, 4))
    plt.imshow(attr_sum, cmap="hot")
    plt.title(title)
    plt.colorbar()
    plt.show()


def compute_expert_attributions(test_trajs, model, env):
    expert_attributions_ig_all = []
    expert_attributions_saliency_all = []

    for traj in test_trajs:
        states = [int(s) for s in traj[:-1]]  # All states except the last one
        destination = int(traj[-1])
        actions = []  # You need to provide the expert actions here

        attributions_ig_traj = []
        attributions_saliency_traj = []

        for idx, state in enumerate(states):
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([destination], dtype=torch.long).to(device)

            # Get the expert action at this step
            action = actions[idx]  # You need to fill in the expert action

            # Get CNN input
            input_data = model.process_features(state_tensor, des_tensor)
            input_data.requires_grad = True

            # Define forward function
            def forward_func(input_data):
                x = input_data
                x = model.forward(x)
                x_mask = model.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs

            predicted_action = torch.tensor([action], dtype=torch.long).to(device)

            # Compute attributions
            ig = IntegratedGradients(forward_func)
            attributions_ig = ig.attribute(input_data, target=predicted_action)

            saliency = Saliency(forward_func)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)

            # Store attributions
            attributions_ig_traj.append(attributions_ig.squeeze().cpu().detach().numpy())
            attributions_saliency_traj.append(attributions_saliency.squeeze().cpu().detach().numpy())

        expert_attributions_ig_all.append(attributions_ig_traj)
        expert_attributions_saliency_all.append(attributions_saliency_traj)

    return expert_attributions_ig_all, expert_attributions_saliency_all


# add new metrics
def compute_entropy(distribution):
    """Compute the entropy of a probability distribution."""
    entropy = -sum(p * log2(p) for p in distribution if p > 0)
    return entropy

def compute_conditional_entropy(trajectory_counts_by_od):
    """Compute the conditional entropy for each OD pair and return the average."""
    total_entropy = 0
    for od_pair, counts in trajectory_counts_by_od.items():
        total = sum(counts.values())
        distribution = [count / total for count in counts.values()]
        entropy = compute_entropy(distribution)
        total_entropy += entropy
    average_entropy = total_entropy / len(trajectory_counts_by_od)
    return average_entropy

def compute_transition_entropy(trajs): #nodes, 1-gram transition
    """Compute the average entropy over the distribution of transitions from nodes."""
    transition_counts = {}
    for traj in trajs:
        for t in range(len(traj) - 1):
            current_node = traj[t]
            next_node = traj[t+1]
            if current_node not in transition_counts:
                transition_counts[current_node] = Counter()
            transition_counts[current_node][next_node] += 1

    entropies = []
    for node, counts in transition_counts.items():
        total = sum(counts.values())
        distribution = [count / total for count in counts.values()]
        entropy = compute_entropy(distribution)
        entropies.append(entropy)

    average_entropy = sum(entropies) / len(entropies) if len(entropies) > 0 else 0
    return average_entropy


def compute_ngram_transition_entropy(trajs, n=2):
    """Compute the average entropy over the distribution of n-gram transitions."""
    ngram_counts = {}
    for traj in trajs:
        for t in range(len(traj) - n):
            ngram = tuple(traj[t:t+n-1])  # Previous n-1 nodes
            next_node = traj[t+n-1]
            if ngram not in ngram_counts:
                ngram_counts[ngram] = Counter()
            ngram_counts[ngram][next_node] += 1

    entropies = []
    for ngram, counts in ngram_counts.items():
        total = sum(counts.values())
        distribution = [count / total for count in counts.values()]
        entropy = compute_entropy(distribution)
        entropies.append(entropy)

    average_entropy = sum(entropies) / len(entropies) if len(entropies) > 0 else 0
    return average_entropy


def compute_simpsons_index(counts):
    """Compute Simpson's Diversity Index."""
    N = sum(counts.values())
    if N == 0:
        return 0
    sum_pi_sq = sum((count / N) ** 2 for count in counts.values())
    sdi = 1 - sum_pi_sq
    inverse_sdi = 1 / sum_pi_sq if sum_pi_sq > 0 else 0
    return sdi, inverse_sdi

# def evaluate_diversity_metrics(expert_trajs, learner_trajs):
#     results = {}

#     # Flatten trajectories to strings
#     expert_trajs_str = ['_'.join(traj) for traj in expert_trajs]
#     learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]

#     # Total number of trajectories
#     total_learner = len(learner_trajs_str)
#     total_expert = len(expert_trajs_str)

#     # Unique trajectories
#     unique_learner = set(learner_trajs_str)
#     unique_expert = set(expert_trajs_str)

#     # 1. Unique Trajectory Counts
#     num_unique_learner = len(unique_learner)
#     proportion_unique_learner = num_unique_learner / total_learner
#     results['Number of Unique Learner Trajectories'] = num_unique_learner
#     results['Proportion of Unique Learner Trajectories'] = proportion_unique_learner

#     # 2. Entropy-Based Measures
#     # Overall Entropy
#     learner_counts = Counter(learner_trajs_str)
#     total_counts = sum(learner_counts.values())
#     learner_distribution = [count / total_counts for count in learner_counts.values()]
#     overall_entropy = compute_entropy(learner_distribution)
#     results['Overall Entropy'] = overall_entropy

#     # Conditional Entropy
#     # Group trajectories by OD pairs
#     learner_od_pairs = [(traj[0], traj[-1]) for traj in learner_trajs]
#     trajectory_counts_by_od = {}
#     for traj_str, od_pair in zip(learner_trajs_str, learner_od_pairs):
#         if od_pair not in trajectory_counts_by_od:
#             trajectory_counts_by_od[od_pair] = Counter()
#         trajectory_counts_by_od[od_pair][traj_str] += 1
#     conditional_entropy = compute_conditional_entropy(trajectory_counts_by_od)
#     results['Average Conditional Entropy'] = conditional_entropy

#     # 3. Diversity Indices
#     sdi, inverse_sdi = compute_simpsons_index(learner_counts)
#     results["Simpson's Diversity Index"] = sdi
#     results["Inverse Simpson's Index"] = inverse_sdi

#     # 4. Novel Trajectory Proportions
#     novel_trajectories = unique_learner - unique_expert
#     num_novel = len(novel_trajectories)
#     proportion_novel = num_novel / num_unique_learner if num_unique_learner > 0 else 0
#     results['Number of Novel Trajectories'] = num_novel
#     results['Proportion of Novel Trajectories'] = proportion_novel

#     return results

def evaluate_diversity_metrics(expert_trajs, learner_trajs):
    results = {}

    # Flatten trajectories to strings
    expert_trajs_str = ['_'.join(traj) for traj in expert_trajs]
    learner_trajs_str = ['_'.join(traj) for traj in learner_trajs]

    # Total number of trajectories
    total_learner = len(learner_trajs_str)
    total_expert = len(expert_trajs_str)

    # Unique trajectories
    unique_learner = set(learner_trajs_str)
    unique_expert = set(expert_trajs_str)

    # 1. Unique Trajectory Counts
    num_unique_learner = len(unique_learner)
    proportion_unique_learner = num_unique_learner / total_learner if total_learner > 0 else 0
    results['Number of Unique Learner Trajectories'] = num_unique_learner
    results['Proportion of Unique Learner Trajectories'] = proportion_unique_learner

    # 2. Entropy-Based Measures
    # Overall Entropy
    learner_counts = Counter(learner_trajs_str)
    total_counts = sum(learner_counts.values())
    learner_distribution = [count / total_counts for count in learner_counts.values()]
    overall_entropy = compute_entropy(learner_distribution)
    results['Overall Entropy'] = overall_entropy

    # Conditional Entropy
    # Group trajectories by OD pairs
    learner_od_pairs = [(traj[0], traj[-1]) for traj in learner_trajs]
    trajectory_counts_by_od = {}
    for traj_str, od_pair in zip(learner_trajs_str, learner_od_pairs):
        if od_pair not in trajectory_counts_by_od:
            trajectory_counts_by_od[od_pair] = Counter()
        trajectory_counts_by_od[od_pair][traj_str] += 1
    conditional_entropy = compute_conditional_entropy(trajectory_counts_by_od)
    results['Average Conditional Entropy'] = conditional_entropy

    # # 3. Diversity Indices
    # sdi, inverse_sdi = compute_simpsons_index(learner_counts)
    # results["Simpson's Diversity Index"] = sdi
    # results["Inverse Simpson's Index"] = inverse_sdi

    # 4. Novel Trajectory Proportions
    novel_trajectories = unique_learner - unique_expert
    num_novel = len(novel_trajectories)
    proportion_novel = num_novel / num_unique_learner if num_unique_learner > 0 else 0
    results['Number of Novel Trajectories'] = num_novel
    results['Proportion of Novel Trajectories'] = proportion_novel

    return results

# no node conditioned entropy
# def evaluate_diversity_metrics_single(trajs, label='Trajectories'):
#     results = {}
    
#     # Flatten trajectories to strings
#     trajs_str = ['_'.join(traj) for traj in trajs]
    
#     # Total number of trajectories
#     total_trajs = len(trajs_str)
    
#     # Unique trajectories
#     unique_trajs = set(trajs_str)
#     num_unique_trajs = len(unique_trajs)
#     proportion_unique_trajs = num_unique_trajs / total_trajs if total_trajs > 0 else 0
#     results[f'Number of Unique {label}'] = num_unique_trajs
#     results[f'Proportion of Unique {label}'] = proportion_unique_trajs
    
#     # Entropy-Based Measures
#     trajs_counts = Counter(trajs_str)
#     total_counts = sum(trajs_counts.values())
#     trajs_distribution = [count / total_counts for count in trajs_counts.values()]
#     overall_entropy = compute_entropy(trajs_distribution)
#     results['Overall Entropy'] = overall_entropy
    
#     # Conditional Entropy
#     # Group trajectories by OD pairs
#     od_pairs = [(traj[0], traj[-1]) for traj in trajs]
#     trajectory_counts_by_od = {}
#     for traj_str, od_pair in zip(trajs_str, od_pairs):
#         if od_pair not in trajectory_counts_by_od:
#             trajectory_counts_by_od[od_pair] = Counter()
#         trajectory_counts_by_od[od_pair][traj_str] += 1
#     conditional_entropy = compute_conditional_entropy(trajectory_counts_by_od)
#     results['Average Conditional Entropy'] = conditional_entropy
    
#     # Diversity Indices
#     sdi, inverse_sdi = compute_simpsons_index(trajs_counts)
#     results["Simpson's Diversity Index"] = sdi
#     results["Inverse Simpson's Index"] = inverse_sdi
    
#     return results



def evaluate_diversity_metrics_single(trajs, label='Trajectories', n=2):
    results = {}
    
    # Flatten trajectories to strings
    trajs_str = ['_'.join(traj) for traj in trajs]
    
    # Total number of trajectories
    total_trajs = len(trajs_str)
    
    # Unique trajectories
    unique_trajs = set(trajs_str)
    num_unique_trajs = len(unique_trajs)
    proportion_unique_trajs = num_unique_trajs / total_trajs if total_trajs > 0 else 0
    results[f'Number of Unique {label}'] = num_unique_trajs
    results[f'Proportion of Unique {label}'] = proportion_unique_trajs
    
    # Entropy-Based Measures
    trajs_counts = Counter(trajs_str)
    total_counts = sum(trajs_counts.values())
    trajs_distribution = [count / total_counts for count in trajs_counts.values()]
    overall_entropy = compute_entropy(trajs_distribution)
    results['Overall Entropy'] = overall_entropy
    
    # Conditional Entropy
    # Group trajectories by OD pairs
    od_pairs = [(traj[0], traj[-1]) for traj in trajs]
    trajectory_counts_by_od = {}
    for traj_str, od_pair in zip(trajs_str, od_pairs):
        if od_pair not in trajectory_counts_by_od:
            trajectory_counts_by_od[od_pair] = Counter()
        trajectory_counts_by_od[od_pair][traj_str] += 1
    conditional_entropy = compute_conditional_entropy(trajectory_counts_by_od)
    results['Average Conditional Entropy'] = conditional_entropy
    
    # # Diversity Indices
    # sdi, inverse_sdi = compute_simpsons_index(trajs_counts)
    # results["Simpson's Diversity Index"] = sdi
    # results["Inverse Simpson's Index"] = inverse_sdi
    
    # Transition Entropy (New Metric)
    transition_entropy = compute_transition_entropy(trajs)
    results['Average Transition Entropy'] = transition_entropy

    # n-gram Transition Entropy (New Metric)
    ngram_transition_entropy = compute_ngram_transition_entropy(trajs, n)
    results[f'Average {n}-gram Transition Entropy'] = ngram_transition_entropy
    
    return results



def evaluate_expert_metrics(expert_trajs):
    return evaluate_diversity_metrics_single(expert_trajs, label='Expert Trajectories', n=2)


def evaluate_model(dataset_name, target_od, target_traj, model, env, n_link=437):
    output_dir = "attributions"  # Folder for saving attributions
    os.makedirs(output_dir, exist_ok=True)
    
    state_ts = torch.from_numpy(np.arange(n_link)).long().to(device)
    target_o, target_d = target_od[:, 0].tolist(), target_od[:, 1].tolist()
    learner_traj = []
    attributions_ig_all = []
    attributions_saliency_all = []
    
    """Compute transition matrix and generate trajectory for each OD pair"""
    for traj_idx, (ori, des) in enumerate(zip(target_o, target_d)):
        # Reset environment for the current origin-destination pair
        curr_ori, curr_des = env.reset(st=ori, des=des)
    
        # Initialize the path and attributions for the current OD pair
        sample_path = [str(curr_ori)]
        curr_state = curr_ori
        attributions_ig_traj = []
        attributions_saliency_traj = []
    
        for _ in range(50):  # Limit the trajectory to 50 steps
            state_tensor = torch.tensor([curr_state], dtype=torch.long).to(device)
            des_tensor = torch.tensor([curr_des], dtype=torch.long).to(device)
    
            # Get action probabilities from the model for the current state and destination
            with torch.no_grad():
                action_probs = model.get_action_prob(state_tensor, des_tensor).squeeze()
                action_probs_np = action_probs.cpu().numpy()
    
            # Choose the most likely action based on the model
            action = np.argmax(action_probs_np)
    
            # Take a step in the environment
            next_state, reward, done = env.step(action)
    
            # Append the next state to the path
            if next_state != env.pad_idx:
                sample_path.append(str(next_state))
            curr_state = next_state
    
            # Compute attributions using Captum
            # Get CNN input
            input_data = model.process_features(state_tensor, des_tensor)
            input_data.requires_grad = True
    
            # Define the forward function for Captum
            def forward_func(input_data):
                x = input_data
                x = model.forward(x)
                x_mask = model.policy_mask[state_tensor]
                x = x.masked_fill((1 - x_mask).bool(), -1e32)
                action_probs = torch.softmax(x, dim=1)
                return action_probs
    
            predicted_action = torch.tensor([action], dtype=torch.long).to(device)
    
            # Compute attributions
            ig = IntegratedGradients(forward_func)
            attributions_ig = ig.attribute(input_data, target=predicted_action)
    
            saliency = Saliency(forward_func)
            attributions_saliency = saliency.attribute(input_data, target=predicted_action)
    
            # Store attributions for this step
            attributions_ig_traj.append(attributions_ig.squeeze().cpu().detach().numpy())
            attributions_saliency_traj.append(attributions_saliency.squeeze().cpu().detach().numpy())
    
            if done:
                break
    
        # Append the generated path and attributions for the current OD pair
        learner_traj.append(sample_path)
        attributions_ig_all.append(attributions_ig_traj)
        attributions_saliency_all.append(attributions_saliency_traj)
    
    # Evaluate metrics for the generated and target trajectories
    edit_dist, bleu_score, js_dist,expert_mean,learner_mean,expert_std,learner_std = evaluate_metrics(target_traj, learner_traj,model)
    
    # # Calculate diversity measures for learner and expert trajectories
    # learner_diversity_results = evaluate_diversity_metrics_single(learner_traj, label='Learner')
    # expert_diversity_results = evaluate_expert_metrics(target_traj)  # Evaluate only the expert diversity
    
    # Calculate diversity measures for learner and expert trajectories
    learner_diversity_results = evaluate_diversity_metrics_single(learner_traj, label=f'{dataset_name} Learner', n=2)
    expert_diversity_results = evaluate_diversity_metrics_single(target_traj, label=f'{dataset_name} Expert', n=2)
    

    # Save trajectory data and attributions to files
    trajectory_data = []
    for idx, (test_path, learner_path, attr_ig_traj, attr_saliency_traj) in enumerate(
        zip(target_traj, learner_traj, attributions_ig_all, attributions_saliency_all)
    ):
        test_traj_str = "_".join(test_path)
        learner_traj_str = "_".join(learner_path)
        trajectory_data.append([test_traj_str, learner_traj_str])
    
        # Save attributions for this trajectory to numpy files in the new folder
        np.save(os.path.join(output_dir, f"{dataset_name}_attributions_ig_traj_{idx}.npy"), attr_ig_traj)
        np.save(os.path.join(output_dir, f"{dataset_name}_attributions_saliency_traj_{idx}.npy"), attr_saliency_traj)
    
        # Save attributions to CSV files for Excel
        # Flatten the attributions for each step and create a DataFrame
        attr_ig_flat = [attr_step.flatten() for attr_step in attr_ig_traj]
        attr_saliency_flat = [attr_step.flatten() for attr_step in attr_saliency_traj]
    
        # Create DataFrames
        df_attr_ig = pd.DataFrame(attr_ig_flat)
        df_attr_saliency = pd.DataFrame(attr_saliency_flat)
    
        # Save to CSV files
        df_attr_ig.to_csv(os.path.join(output_dir, f"{dataset_name}_attributions_ig_traj_{idx}.csv"), index=False)
        df_attr_saliency.to_csv(os.path.join(output_dir, f"{dataset_name}_attributions_saliency_traj_{idx}.csv"), index=False)
    
        # Save to Excel files
        df_attr_ig.to_excel(os.path.join(output_dir, f"{dataset_name}_attributions_ig_traj_{idx}.xlsx"), index=False)
        df_attr_saliency.to_excel(os.path.join(output_dir, f"{dataset_name}_attributions_saliency_traj_{idx}.xlsx"), index=False)
    
    # Save trajectory data to CSV with the dataset name as prefix
    with open(f"{dataset_name}_trajectory_with_timestep.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Expert Trajectory", "Learner Trajectory"])
        csv_writer.writerows(trajectory_data)
    
    
    # Print evaluation metrics
    print(f"{dataset_name} - edit dist {edit_dist}")
    print(f"{dataset_name} - bleu score {bleu_score}")
    print(f"{dataset_name} - js distance {js_dist}")
    # print(f"{dataset_name} - avg_log_likeli {avg_log_likeli}")
    print(f"{dataset_name} - expert_mean {expert_mean}, earner_mean {learner_mean}")
    print(f"{dataset_name} - expert_std {expert_std}, learner_std{learner_std}")
    
    # Print diversity metrics for learner trajectories
    print(f"\n{dataset_name} Learner Diversity Metrics:")
    for key, value in learner_diversity_results.items():
        print(f"{key}: {value}")
    
    # Print diversity metrics for expert trajectories
    print("\nExpert Diversity Metrics:")
    for key, value in expert_diversity_results.items():
        print(f"{key}: {value}")

    return edit_dist, bleu_score, js_dist, learner_diversity_results, expert_diversity_results

