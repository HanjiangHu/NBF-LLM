from train import *


def find_emb_ssm_nbf(model_path, emb_name='all-mpnet-base-v2',state_dim=768, input_dim=768,output_dim=768,hidden_dim=512, hidden_dim_nbf=32,class_num=5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(emb_name)  

    ssm = NeuralStateSpaceModel(state_dim, input_dim, output_dim, hidden_dim)
    nbf = NeuralBarrierFunction(state_dim, input_dim, hidden_dim_nbf,class_num=class_num)
    ssm.load_state_dict(torch.load(model_path)['ssm'])
    ssm.to(device) 
    nbf.load_state_dict(torch.load(model_path)['nbf'])
    nbf.to(device)
    return model, ssm, nbf, device

def calculate_score(queries:list, i:int,model, ssm, nbf, device):
    assert (i>=0) and (i<len(queries))
    state_dim = ssm.state_transition[-1].out_features
    # Initialize hidden states (x0) to zeros
    x_t = torch.zeros(1, state_dim, device=device)
    for ind, query in enumerate(queries):
        x_t_prev = x_t.clone()
        u_t = model.encode(query, convert_to_tensor=True).unsqueeze(dim=0).to(device)
        
        nbf_output = nbf(x_t_prev, u_t)

        x_t, y_t_pred = ssm(x_t_prev, u_t)
        probs = torch.softmax(nbf_output, dim=-1)  # Convert logits to probabilities
        last_class_prob = probs[:, -1]  # Probability of the last class
        max_other_class_prob = torch.max(probs[:, :-1], dim=1).values  # Max prob of other classes
        nbf_score = last_class_prob - max_other_class_prob
        if ind == i:
            return nbf_score[0].item() 

def calculate_score_from_dialog(dialog_hist, summary_query,model, ssm, nbf, device):
    state_dim = ssm.state_transition[-1].out_features
    # Initialize hidden states (x0) to zeros
    x_t = torch.zeros(1, state_dim, device=device)
    for dialog in dialog_hist:
        if dialog['role'] == 'user':
            x_t_prev = x_t.clone()
            u_t = model.encode(dialog['content'], convert_to_tensor=True).unsqueeze(dim=0).to(device)
            nbf_output = nbf(x_t_prev, u_t)
            x_t, y_t_pred = ssm(x_t_prev, u_t)
    u_t = model.encode(summary_query, convert_to_tensor=True).unsqueeze(dim=0).to(device)
    nbf_output = nbf(x_t, u_t)
    x_t, y_t_pred = ssm(x_t, u_t)
    probs = torch.softmax(nbf_output, dim=-1)  # Convert logits to probabilities
    last_class_prob = probs[:, -1]  # Probability of the last class
    max_other_class_prob = torch.max(probs[:, :-1], dim=1).values  # Max prob of other 4 classes
    nbf_score = last_class_prob - max_other_class_prob
    return nbf_score[0].item() 

