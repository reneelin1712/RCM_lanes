import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import csv
from model.policy import PolicyCNN
from model.value import ValueCNN
from model.discriminator import DiscriminatorAIRLCNN
from network_env import RoadWorld
from utils.torch import to_device
from utils.load_data import ini_od_dist, load_path_feature, load_link_feature, minmax_normalization

def load_model(model_path, policy_net, value_net, discrim_net):
    model_dict = torch.load(model_path)
    policy_net.load_state_dict(model_dict['Policy'])
    print("Policy Model loaded Successfully")
    value_net.load_state_dict(model_dict['Value'])
    print("Value Model loaded Successfully")
    discrim_net.load_state_dict(model_dict['Discrim'])
    print("Discrim Model loaded Successfully")

def evaluate_rewards(traj_data, policy_net, discrim_net, env, transit_dict):
    device = torch.device('cpu')  # Use CPU device
    policy_net.to(device)
    discrim_net.to(device)
    
    reward_data = []
    input_features = []
    output_rewards = []
    
    for episode_idx, traj in enumerate(traj_data):
        path = traj.split('_')
        
        des = torch.LongTensor([int(path[-1])]).to(device)
        
        for step_idx in range(len(path) - 1):
            state = torch.LongTensor([int(path[step_idx])]).to(device)
            next_state = torch.LongTensor([int(path[step_idx + 1])]).to(device)
            
            action = transit_dict.get((int(path[step_idx]), int(path[step_idx + 1])), 'N/A')
            action_tensor = torch.LongTensor([action]).to(device) if action != 'N/A' else None
            
            if action_tensor is not None:
                with torch.no_grad():
                    neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, next_path_feature, next_edge_feature = discrim_net.get_input_features(state, des, action_tensor, next_state)
                    log_prob = policy_net.get_log_prob(state, des, action_tensor).squeeze()
                    reward = discrim_net.forward_with_actual_features(neigh_path_feature, neigh_edge_feature, path_feature, edge_feature, action_tensor, log_prob, next_path_feature, next_edge_feature)
                    
                    # Collect input features
                    input_feature = {
                        'neigh_path_feature': neigh_path_feature.cpu().numpy().flatten().tolist(),
                        'neigh_edge_feature': neigh_edge_feature.cpu().numpy().flatten().tolist(),
                        'path_feature': path_feature.cpu().numpy().flatten().tolist(),
                        'edge_feature': edge_feature.cpu().numpy().flatten().tolist(),
                        'next_path_feature': next_path_feature.cpu().numpy().flatten().tolist(),
                        'next_edge_feature': next_edge_feature.cpu().numpy().flatten().tolist(),
                        'action': action_tensor.item(),
                        'log_prob': log_prob.item()
                    }
                    input_features.append(input_feature)
                    output_rewards.append(reward.item())
            else:
                reward = torch.tensor('N/A')
            
            reward_data.append({
                'episode': episode_idx,
                'des': des.item(),
                'step': step_idx,
                'state': path[step_idx],
                'action': action,
                'next_state': path[step_idx + 1],
                'reward': reward.item() if reward != 'N/A' else 'N/A'
            })
    
    # Convert reward_data to a pandas DataFrame
    reward_df = pd.DataFrame(reward_data)
    
    return reward_df, input_features, output_rewards

def create_shap_explainer(model, input_features):
    feature_keys = ['neigh_path_feature', 'neigh_edge_feature', 'path_feature', 'edge_feature', 'next_path_feature', 'next_edge_feature', 'action', 'log_prob']
    
    flattened_features = []
    
    for feature_dict in input_features:
        sample_features = []
        for key in feature_keys:
            if key in feature_dict:
                if isinstance(feature_dict[key], (int, float)):
                    sample_features.append(feature_dict[key])
                else:
                    sample_features.extend(np.array(feature_dict[key]).flatten())
            else:
                print(f"Warning: {key} not found in feature dictionary")
        flattened_features.append(sample_features)
    
    input_features_array = np.array(flattened_features)

    def predict_fn(input_features):
        num_samples = input_features.shape[0]
        model_outputs = np.zeros(num_samples)

        for i in range(num_samples):
            neigh_path_feature = torch.tensor(input_features[i, :108].reshape(9, 12), dtype=torch.float32)
            neigh_edge_feature = torch.tensor(input_features[i, 108:180].reshape(9, 8), dtype=torch.float32)
            path_feature = torch.tensor(input_features[i, 180:192], dtype=torch.float32)
            edge_feature = torch.tensor(input_features[i, 192:200], dtype=torch.float32)
            action = torch.tensor(input_features[i, 200], dtype=torch.long)
            log_prob = torch.tensor(input_features[i, 201], dtype=torch.float32)
            next_path_feature = torch.tensor(input_features[i, 202:214], dtype=torch.float32)
            next_edge_feature = torch.tensor(input_features[i, 214:222], dtype=torch.float32)

            action = torch.clamp(action, min=0, max=model.action_num - 1)

            model_output = model.forward_with_actual_features(
                neigh_path_feature.unsqueeze(0),
                neigh_edge_feature.unsqueeze(0),
                path_feature.unsqueeze(0),
                edge_feature.unsqueeze(0),
                action.unsqueeze(0),
                log_prob.unsqueeze(0),
                next_path_feature.unsqueeze(0),
                next_edge_feature.unsqueeze(0)
            ).detach().numpy()

            model_outputs[i] = model_output

        return model_outputs

    background_data = shap.sample(input_features_array, 10) 
    explainer = shap.KernelExplainer(predict_fn, background_data)
    return explainer

def analyze_shap_values(explainer, input_features, feature_indices):
    feature_keys = ['neigh_path_feature', 'neigh_edge_feature', 'path_feature', 'edge_feature', 'next_path_feature', 'next_edge_feature', 'action', 'log_prob']
    
    flattened_features = []
    
    for feature_dict in input_features:
        sample_features = []
        for key in feature_keys:
            if key in feature_dict:
                if isinstance(feature_dict[key], (int, float)):
                    sample_features.append(feature_dict[key])
                else:
                    sample_features.extend(np.array(feature_dict[key]).flatten())
            else:
                print(f"Warning: {key} not found in feature dictionary")
        flattened_features.append(sample_features)
    
    input_features_array = np.array(flattened_features)

    shap_values = explainer.shap_values(input_features_array)
    shap_values_squeezed = np.squeeze(shap_values)

    selected_shap_values = shap_values_squeezed[:, feature_indices]

    feature_names_dict = {
        180: 'shortest_distance', 181: 'number_of_links', 182: 'number_of_left_turn',
        183: 'number_of_right_turn', 184: 'number_of_u_turn', 185: 'freq_road_type_1',
        186: 'freq_road_type_2', 187: 'freq_road_type_3', 188: 'freq_road_type_4',
        189: 'freq_road_type_5', 190: 'freq_road_type_6', 191: 'link_length',
        192: 'lanes', 193: 'road_type_1', 194: 'road_type_2', 195: 'road_type_3',
        196: 'road_type_4', 197: 'road_type_5', 198: 'road_type_6'
    }
    selected_feature_names = [feature_names_dict.get(index, f'Feature {index}') for index in feature_indices]

    selected_shap_values_df = pd.DataFrame(selected_shap_values, columns=selected_feature_names)
    selected_shap_values_df.to_csv('selected_shap_values.csv', index=False)
    print("Selected SHAP values saved to 'selected_shap_values.csv'")

    shap.summary_plot(selected_shap_values, input_features_array[:, feature_indices], plot_type="bar", feature_names=selected_feature_names, show=False)
    plt.savefig('selected_shap_summary_plot.png')
    plt.close()
    print("Selected SHAP summary plot saved to 'selected_shap_summary_plot.png'")

if __name__ == "__main__":
    # Configuration
    cv = 0
    size = 10000
    gamma = 0.99
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    edge_p = "../data/base/edge.txt"
    network_p = "../data/base/transit.npy"
    path_feature_p = "../data/base/feature_od.npy"
    train_p = "../data/base/cross_validation/train_CV%d_size%d.csv" % (cv, size)
    test_p = "../data/base/cross_validation/test_CV%d.csv" % cv
    model_p = "../trained_models/base/airl_CV%d_size%d.pt" % (cv, size)


    # Initialize environment
    od_list, od_dist = ini_od_dist(train_p)
    env = RoadWorld(network_p, edge_p, pre_reset=(od_list, od_dist))

    # Load features
    path_feature, path_max, path_min = load_path_feature(path_feature_p)
    edge_feature, link_max, link_min = load_link_feature(edge_p)
    path_feature = minmax_normalization(path_feature, path_max, path_min)
    path_feature_pad = np.zeros((env.n_states, env.n_states, path_feature.shape[2]))
    path_feature_pad[:path_feature.shape[0], :path_feature.shape[1], :] = path_feature
    edge_feature = minmax_normalization(edge_feature, link_max, link_min)
    edge_feature_pad = np.zeros((env.n_states, edge_feature.shape[1]))
    edge_feature_pad[:edge_feature.shape[0], :] = edge_feature

    # Initialize models
    policy_net = PolicyCNN(env.n_actions, env.policy_mask, env.state_action,
                           path_feature_pad, edge_feature_pad,
                           path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                           env.pad_idx).to(device)
    value_net = ValueCNN(path_feature_pad, edge_feature_pad,
                         path_feature_pad.shape[-1] + edge_feature_pad.shape[-1]).to(device)
    discrim_net = DiscriminatorAIRLCNN(env.n_actions, gamma, env.policy_mask,
                                       env.state_action, path_feature_pad, edge_feature_pad,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1] + 1,
                                       path_feature_pad.shape[-1] + edge_feature_pad.shape[-1],
                                       env.pad_idx).to(device)

    # Load trained models
    load_model(model_p, policy_net, value_net, discrim_net)

    # Load transit data
    transit_data = pd.read_csv('../data/base/transit.csv')
    transit_dict = {(row['link_id'], row['next_link_id']): row['action'] for _, row in transit_data.iterrows()}

    # Load trajectory data
    trajectory_data = []
    with open('trajectories.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            trajectory_data.append(row[0])  # Assuming the trajectory is in the first column

    # Evaluate rewards
    reward_df, input_features, output_rewards = evaluate_rewards(trajectory_data, policy_net, discrim_net, env, transit_dict)
    print('input_features length:', len(input_features))

    # Create SHAP explainer
    explainer = create_shap_explainer(discrim_net, input_features)

    # Define feature indices for analysis
    selected_feature_indices = [
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,  # Path features
        192, 193, 194, 195, 196, 197, 198  # Edge features
    ]

    # Analyze SHAP values
    analyze_shap_values(explainer, input_features[:10], selected_feature_indices)