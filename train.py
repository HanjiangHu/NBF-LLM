from sentence_transformers import SentenceTransformer
import torch, os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse

model = SentenceTransformer('all-mpnet-base-v2') 

import json

class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.max_length = max(len(u) for u in trajectories)

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        user_embeddings = torch.stack([step["u"] for step in trajectory])
        assistant_embeddings = torch.stack([step["y"] for step in trajectory])
        scores = torch.tensor([step["score"] for step in trajectory])

        # Pad sequences to max_length
        u_padded = torch.zeros(self.max_length, user_embeddings.size(1))
        y_padded = torch.zeros(self.max_length, assistant_embeddings.size(1))
        score_padded = torch.zeros(self.max_length)
        mask = torch.zeros(self.max_length)

        length = len(trajectory)
        u_padded[:length] = user_embeddings
        y_padded[:length] = assistant_embeddings
        score_padded[:length] = scores - 1
        mask[:length] = 1  # Mask indicating valid time steps

        return u_padded, y_padded, score_padded, mask


class NeuralStateSpaceModel(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, hidden_dim):
        super(NeuralStateSpaceModel, self).__init__()
        # Non-linear state transition (f)
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

       
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x_t_pre, u_t):
        """
        Forward pass through the state-space model.
        
        Args:
            x_t (torch.Tensor): State vector at time t (shape: [batch_size, state_dim]).
            u_t (torch.Tensor): Input vector at time t (shape: [batch_size, input_dim]).
        
        Returns:
            x_t_next (torch.Tensor): Predicted state vector at time t+1.
            y_t (torch.Tensor): Observed output vector at time t.
        """
        # Combine state and input for state transition
        xu_t_pre = torch.cat([x_t_pre, u_t], dim=-1)

        # State transition
        x_t = self.state_transition(xu_t_pre)

        xu_t = torch.cat([x_t, u_t], dim=-1)
        # Observation
        y_t = self.observation_model(xu_t)

        return x_t, y_t


def loss_forward_invariance(cbf_next, FI_mask, label=None):
    probs = torch.softmax(cbf_next, dim=-1)  # Convert logits to probabilities
    last_class_prob = probs[:,:, -1]  # Probability of the last class
    max_other_class_prob = torch.max(probs[:,:, :-1], dim=2).values  # Max prob of other 4 classes
    if label is None:
        loss = torch.relu(last_class_prob - max_other_class_prob) * FI_mask[:,:, 0]
    else:
        safe_mask = (label[:,:, 0]!=probs.shape[-1]-1)
        loss = torch.relu((2 * safe_mask - 1) * (last_class_prob - max_other_class_prob)) * FI_mask[:,:, 0]
    if FI_mask.sum() == 0: return torch.tensor(0.0)
    return loss.sum() / FI_mask.sum()


class NeuralBarrierFunction(nn.Module):
    """
    Neural Barrier Function (NBF) model.
    """
    def __init__(self, state_dim, input_dim, hidden_dim,class_num=5):
        super(NeuralBarrierFunction, self).__init__()
        self.nbf = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, class_num), 
        )

    def forward(self, x, u):
        xu = torch.cat([x, u], dim=-1)  # Concatenate state and input
        return self.nbf(xu)

def train_dialogue_dynamics(ssm, dataset, num_epochs=200, batch_size=64, validation_split=0.05, save_path="ssm_model.pth", val_dataset_hb=None, weight_decay=0,ssm_learning_rate=1e-4): # TODO: weight_nbf needs to be tuned
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssm.to(device)
    if val_dataset_hb is None:
        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        # use Harmbench data as val
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_hb, batch_size=batch_size, shuffle=False)

    # Initialize optimizers for SSM
    optimizer_ssm = optim.Adam(ssm.parameters(), lr=ssm_learning_rate, weight_decay=weight_decay)

    loss_fn_mse = nn.MSELoss()  # Loss for SSM     
    writer = SummaryWriter(log_dir=save_path)
    best_val_loss_ssm = float('inf')  # For saving the best model

    for epoch in range(num_epochs):
        # Training
        ssm.train()
        total_ssm_loss = 0.0

        for u_batch, y_batch, score_batch, mask in train_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            score_batch = score_batch.unsqueeze(-1).to(device)
            mask = mask.unsqueeze(-1).to(device)
            batch_size, seq_len, state_dim = u_batch.shape

            # Initialize hidden states (x0) to zeros
            x_t = torch.zeros(batch_size, state_dim, device=device)

            # Forward pass through the SSM
            predicted_y = []
            for t in range(seq_len):
                u_t = u_batch[:, t, :]
                # SSM forward pass
                x_t, y_t = ssm(x_t, u_t)
                predicted_y.append(y_t)
            
            # Stack predicted outputs along the time dimension
            predicted_y = torch.stack(predicted_y, dim=1)

            ssm_loss = loss_fn_mse(predicted_y * mask, y_batch * mask)
            optimizer_ssm.zero_grad()
            ssm_loss.backward() 
            optimizer_ssm.step()

            total_ssm_loss += ssm_loss.item()
        
        # Validation
        ssm.eval()
        val_total_ssm_loss = 0.0
        with torch.no_grad():
            for u_batch, y_batch, score_batch, mask in val_loader:
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                score_batch = score_batch.unsqueeze(-1).to(device)
                mask = mask.unsqueeze(-1).to(device)
                batch_size, seq_len, state_dim = u_batch.shape

                # Initialize hidden states (x0) to zeros
                x_t = torch.zeros(batch_size, state_dim, device=device)

                # Forward pass through the SSM
                predicted_y = []
                for t in range(seq_len):
                    u_t = u_batch[:, t, :]
                    # SSM forward pass
                    x_t, y_t = ssm(x_t, u_t)
                    predicted_y.append(y_t)

                # Stack predicted outputs along the time dimension
                predicted_y = torch.stack(predicted_y, dim=1)

                ssm_loss = loss_fn_mse(predicted_y * mask, y_batch * mask)

                val_total_ssm_loss += ssm_loss.item()
                
        # Average losses
        avg_train_ssm_loss = total_ssm_loss / len(train_loader)
        avg_val_total_ssm_loss = val_total_ssm_loss / len(val_loader)

        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train_SSM", avg_train_ssm_loss, epoch)
        writer.add_scalar("Loss/Val_SSM", avg_val_total_ssm_loss, epoch)


        print(f"Epoch {epoch + 1}/{num_epochs}"
              f"Train SSM Loss: {avg_train_ssm_loss:.4f}"
              f"Val SSM Loss: {avg_val_total_ssm_loss:.4f}")

        # Save the model if validation loss improves
        if avg_val_total_ssm_loss < best_val_loss_ssm:
            best_val_loss_ssm = avg_val_total_ssm_loss
            torch.save({
                'ssm': ssm.state_dict(),
            }, f"{save_path}/models_best_ssm.pth")
            print(f"Models saved as models_best_ssm, ssm loss: {avg_val_total_ssm_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'ssm': ssm.state_dict(),
            }, save_path+  f"/ssm_model_epoch{epoch + 1}.pth")


# Training and validation function
def train_barrier_function(ssm, nbf, dataset, num_epochs=200, nbf_learning_rate=1e-3, batch_size=64, validation_split=0.05, save_path="ssm_model.pth", val_dataset_hb=None, ssm_model_path=None,  weight_FI=100, ROA_interval=3,weight_nbf=1,weight_decay=0,weight_safe=100): # TODO: weight_nbf needs to be tuned
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssm.to(device)
    nbf.to(device)
    if val_dataset_hb is None:
        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset_hb, batch_size=batch_size, shuffle=False)

    # Initialize optimizer for NBF
    optimizer = optim.Adam(nbf.parameters(), lr=nbf_learning_rate, weight_decay=weight_decay)
    ssm.load_state_dict(torch.load(ssm_model_path)['ssm']) 
    ssm.to(device)

    loss_fn_ce = nn.CrossEntropyLoss()        
    writer = SummaryWriter(log_dir=save_path)
    best_val_loss_nbf = float('inf')  # For saving the best model


    for epoch in range(num_epochs):
        # Training
        ssm.train()
        nbf.train()
        total_nbf_loss = 0.0
        total_nbf_ce_loss = 0.0
        total_FI_loss = 0.0
        total_safe_loss = 0.0
        for u_batch, y_batch, score_batch, mask in train_loader:
            u_batch = u_batch.to(device)
            y_batch = y_batch.to(device)
            score_batch = score_batch.unsqueeze(-1).to(device)
            mask = mask.unsqueeze(-1).to(device)
            batch_size, seq_len, state_dim = u_batch.shape

            # Initialize hidden states (x0) to zeros
            x_t = torch.zeros(batch_size, state_dim, device=device)

            # Forward pass through the SSM
            nbf_predictions = []
            nbf_predictions_next = []
            nbf_labels = []
            # print(mask[0])
            for t in range(seq_len):
                u_t = u_batch[:, t, :]
                x_t_prev = x_t.clone()
                # SSM forward pass
                x_t, y_t = ssm(x_t, u_t)

                mask_forward_invariance = torch.zeros_like(mask, device=device)
                mask_forward_invariance[:, :-ROA_interval, :] = mask[:,ROA_interval:,:]

                # assume the adversarial u is also adversarial for the next state
                u_t_next = u_batch[:, t+1, :] if t < seq_len - 1 else u_t.clone()
                # NBF forward pass
                nbf_output = nbf(x_t_prev, u_t)
                nbf_output_next = nbf(x_t, u_t_next)

                nbf_predictions.append(nbf_output)
                nbf_predictions_next.append(nbf_output_next)
                nbf_labels.append(score_batch[:, t, :])
            
            # Stack predicted outputs along the time dimension
            nbf_predictions = torch.stack(nbf_predictions, dim=1)  # NBF predictions

            nbf_predictions_next = torch.stack(nbf_predictions_next, dim=1)  # NBF predictions for FI
            nbf_labels = torch.stack(nbf_labels, dim=1)

            
            masked_nbf_pred = nbf_predictions * mask
            masked_nbf_labels = nbf_labels * mask
            nbf_ce_loss = loss_fn_ce(masked_nbf_pred.view(-1, masked_nbf_pred.size(-1)), masked_nbf_labels.to(torch.int64).view(-1))
            forward_invariance_loss = loss_forward_invariance(nbf_predictions_next, mask_forward_invariance)
            safe_set_loss = loss_forward_invariance(nbf_predictions, mask,nbf_labels)
            nbf_total_loss = nbf_ce_loss + weight_FI * forward_invariance_loss + weight_safe * safe_set_loss

            total_loss = weight_nbf * nbf_total_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_nbf_loss += nbf_total_loss.item()

            total_nbf_ce_loss += nbf_ce_loss.item()
            total_FI_loss += forward_invariance_loss.item()
            total_safe_loss += safe_set_loss.item()
        
        # Validation
        ssm.eval()
        nbf.eval()
        val_total_nbf_loss = 0.0
        val_total_nbf_ce_loss = 0.0
        val_total_FI_loss = 0.0
        val_total_loss_barrier_score = 0.0
        with torch.no_grad():
            for u_batch, y_batch, score_batch, mask in val_loader:
                u_batch = u_batch.to(device)
                y_batch = y_batch.to(device)
                score_batch = score_batch.unsqueeze(-1).to(device)
                mask = mask.unsqueeze(-1).to(device)
                batch_size, seq_len, state_dim = u_batch.shape

                # Initialize hidden states (x0) to zeros
                x_t = torch.zeros(batch_size, state_dim, device=device)

                # Forward pass through the SSM
                nbf_predictions = []
                nbf_predictions_next = []
                nbf_labels = []
                for t in range(seq_len):
                    u_t = u_batch[:, t, :]
                    x_t_prev = x_t.clone()
                    # SSM forward pass
                    x_t, y_t = ssm(x_t, u_t)

                    mask_forward_invariance = torch.zeros_like(mask, device=device)
                    mask_forward_invariance[:, :-ROA_interval, :] = mask[:,ROA_interval:,:]

                    # assume the adversarial u is also adversarial for the next state
                    u_t_next = u_batch[:, t+1, :] if t < seq_len - 1 else u_t.clone()
                    # NBF forward pass
                    nbf_output = nbf(x_t_prev.clone().detach(), u_t)
                    nbf_output_next = nbf(x_t.clone().detach(), u_t_next)
                    
                    nbf_predictions.append(nbf_output)
                    nbf_predictions_next.append(nbf_output_next)
                    nbf_labels.append(score_batch[:, t, :])

                
                # Stack predicted outputs along the time dimension

                nbf_predictions = torch.stack(nbf_predictions, dim=1)  # NBF predictions

                nbf_predictions_next = torch.stack(nbf_predictions_next, dim=1)  # NBF predictions for FI
                nbf_labels = torch.stack(nbf_labels, dim=1)
                
                masked_nbf_pred = nbf_predictions * mask
                masked_nbf_labels = nbf_labels * mask
                nbf_ce_loss = loss_fn_ce(masked_nbf_pred.view(-1, masked_nbf_pred.size(-1)), masked_nbf_labels.to(torch.int64).view(-1))
                forward_invariance_loss = loss_forward_invariance(nbf_predictions_next, mask_forward_invariance)
                safe_set_loss = loss_forward_invariance(nbf_predictions, mask,nbf_labels)
                nbf_total_loss = nbf_ce_loss + weight_FI * forward_invariance_loss + weight_safe * safe_set_loss
            
                total_loss = weight_nbf * nbf_total_loss

                val_total_nbf_loss += nbf_total_loss.item()

                val_total_nbf_ce_loss += nbf_ce_loss.item()
                val_total_FI_loss += forward_invariance_loss.item()

                val_total_loss_barrier_score += safe_set_loss.item() 

        # Average losses
        avg_train_nbf_loss = total_nbf_loss / len(train_loader)
        avg_nbf_ce_loss = total_nbf_ce_loss / len(train_loader)
        avg_FI_loss= total_FI_loss / len(train_loader)  
        avg_safe_loss = total_safe_loss / len(train_loader)  
        
        avg_val_total_nbf_loss = val_total_nbf_loss / len(val_loader)
        avg_val_total_nbf_ce_loss= val_total_nbf_ce_loss / len(val_loader)
        avg_val_total_FI_loss= val_total_FI_loss / len(val_loader) 
        
        avg_val_total_loss_barrier_score = val_total_loss_barrier_score / len(val_loader) 
        

        # Log losses to TensorBoard
        writer.add_scalar("Loss/Train_NBF", avg_train_nbf_loss, epoch)
        writer.add_scalar("Loss/Train_NBF_safe_ce", avg_nbf_ce_loss, epoch)
        writer.add_scalar("Loss/Train_NBF_FI", avg_FI_loss, epoch)
        writer.add_scalar("Loss/Train_NBF_safe_barrier_loss", avg_safe_loss, epoch)
        
        writer.add_scalar("Loss/Val_NBF", avg_val_total_nbf_loss, epoch)
        writer.add_scalar("Loss/Val_NBF_safe_ce", avg_val_total_nbf_ce_loss, epoch)
        writer.add_scalar("Loss/Val_NBF_FI", avg_val_total_FI_loss, epoch)
        writer.add_scalar("Loss/val_barrier_loss", avg_val_total_loss_barrier_score, epoch)


        print(f"Epoch {epoch + 1}/{num_epochs}"
              f"Train NBF Loss: {avg_train_nbf_loss:.4f}, safe: {avg_nbf_ce_loss:.4f}, FI: {avg_FI_loss:.4f}, safe: {avg_safe_loss:.4f}"
              f"Val NBF Loss: {avg_val_total_nbf_loss:.4f}, safe: {avg_val_total_nbf_ce_loss:.4f}, FI: {avg_val_total_FI_loss:.4f}, safe: {avg_val_total_loss_barrier_score:.4f}")

        # Save the model if validation loss improves
        if avg_val_total_nbf_ce_loss < best_val_loss_nbf:
            best_val_loss_nbf = avg_val_total_nbf_ce_loss
            torch.save({
                'ssm': ssm.state_dict(),
                'nbf': nbf.state_dict()
            }, f"{save_path}/models_best_nbf.pth")
            print(f"Models saved as models_best_nbf, nbf cross-entropy loss: {avg_val_total_nbf_ce_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            torch.save({
                'ssm': ssm.state_dict(),
                'nbf': nbf.state_dict()
            }, save_path+  f"/nbf_model_epoch{epoch + 1}.pth")

def read_json_files_in_folder(folder_path, min_turn=1,remove_refusal=True):
    conversations = [] 
    # Traverse through the folder
    for root, dirs, files in os.walk(folder_path):
        for index, file in enumerate(files):
            file_path = os.path.join(root, file)
            if "actorattack" in file_path: continue
            print("loading data from attack method: ",index,file_path)
             # To store structured conversation data
            current_conversation = []  # To track each conversation
            # Open and read the file line by line
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Parse the JSON line
                    data = json.loads(line.strip())
                    assert "actorattack" not in file_path
                    if "goal_achieved" in data:
                        if len(current_conversation) >= min_turn:
                                # ending the last conversation
                                conversations.append(current_conversation)
                        current_conversation = []
                    if "round" in data:
                        if remove_refusal:
                            if data['score'] != 'refused':
                                assert data['score'] > 0
                                current_conversation.append({'u':model.encode(data['user'], convert_to_tensor=True),
                                                            'y':model.encode(data['assistant'], convert_to_tensor=True),
                                                            'score':data['score'],
                                                            'round': data['round']
                                                            })
                        else:
                            if data['score'] == 'refused': data['score'] = 0
                            current_conversation.append({'u':model.encode(data['user'], convert_to_tensor=True),
                                                        'y':model.encode(data['assistant'], convert_to_tensor=True),
                                                        'score':data['score'],
                                                        'round': data['round']
                                                        })       
    return conversations

def transform_actorattack_data(file_path, save_path=None):
    # find the embeddings from ActorAttack results, https://github.com/AI45Lab/ActorAttack
    with open(file_path, "r") as file:
        data = json.load(file)
    # Initialize a list to hold all dialog trajectories
    dialog_trajectories = []
    # Iterate through the `data` field
    for item in data["data"]:
        # Access the dialog history for each attempt
        for attempt in item["attempts"]:
            dialog_hist = attempt["dialog_hist"]
            # Extract user ("u"), assistant ("y"), and score for each step
            assistant_flag = False
            diaglog_item = []
            for step in dialog_hist:
                if step["role"] == "user":
                    assert not assistant_flag
                    user_content = step["content"] 
                else:
                    assistant_content = step["content"] 
                    score = step["score"] 
                    assistant_flag = not assistant_flag
                # Add only user-assistant pairs (ignore None)
                if step["role"] == "assistant":
                    assert assistant_flag
                    diaglog_item.append({
                        "u": model.encode(user_content, convert_to_tensor=True),
                        "y": model.encode(assistant_content, convert_to_tensor=True),
                        "score": score
                    })
                    assistant_flag = not assistant_flag
            dialog_trajectories.append(diaglog_item)
    if save_path is not None: torch.save(dialog_trajectories, save_path)
    return dialog_trajectories



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./models",help="Path to save models")
    parser.add_argument("--find_embedding", action='store_true', help=" Whether to find embeddings from dialogues for model training, default False")
    args = parser.parse_args()
       

    state_dim = 768    # Latent state dimension
    input_dim = 768    # Input embedding dimension
    output_dim = 768   # Observation embedding dimension
    hidden_dim_ssm = 512  # Hidden layer size of dialogue dynamics
    hidden_dim_nbf = 32 # Hidden layer size of barrier function
    find_embedding = args.find_embedding

    all_dialog_trajectories = []
    cb_actorattack = []
    val_hb_dialog_trajectories_all = []
    val_hb_dialog_trajectories_actorattack = []
    
    # load actorattack training data from circuit breakers 1k training data
    if find_embedding:
        cb_actorattack = transform_actorattack_data(file_path="./data/train/actorattack_1k_cb.json",save_path="circuit_breakers_actorattack.pt")
        all_dialog_trajectories = read_json_files_in_folder("./data/train/")
        torch.save(all_dialog_trajectories, "circuit_breakers_others.pt")
    else:
         cb_actorattack = torch.load("circuit_breakers_actorattack.pt")
         all_dialog_trajectories = torch.load("circuit_breakers_others.pt") 
    dataset = TrajectoryDataset(all_dialog_trajectories+cb_actorattack)
    
    # build val data from harmbench
    if find_embedding:
        val_hb_dialog_trajectories_all = read_json_files_in_folder("./data/val/")
        val_hb_dialog_trajectories_actorattack = transform_actorattack_data(file_path="./data/val/actorattack_200_hb.json",save_path="harmbench_actorattack.pt")
        torch.save(val_hb_dialog_trajectories_all, "harmbench_others.pt")
    else:
        val_hb_dialog_trajectories_all = torch.load("harmbench_others.pt")
        val_hb_dialog_trajectories_actorattack = torch.load("harmbench_actorattack.pt")
    val_dataset = TrajectoryDataset(val_hb_dialog_trajectories_all+val_hb_dialog_trajectories_actorattack)

    
    save_path = args.save_path + "/default"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # initial ssm and nbf
    ssm = NeuralStateSpaceModel(state_dim, input_dim, output_dim, hidden_dim_ssm)
    nbf = NeuralBarrierFunction(state_dim, input_dim, hidden_dim_nbf)
    
    # train ssm
    train_dialogue_dynamics(ssm, dataset, num_epochs=200, batch_size=64, save_path=save_path,val_dataset_hb=val_dataset)
    
    # train nbf 
    train_barrier_function(ssm, nbf, dataset, num_epochs=200, nbf_learning_rate=1e-3, batch_size=64, save_path=save_path,val_dataset_hb=val_dataset,ssm_model_path=f"{save_path}/models_best_ssm.pth",weight_FI=100,ROA_interval=3,weight_safe=100)

    
    



    

