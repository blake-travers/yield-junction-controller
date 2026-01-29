import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = 3

class TrafficGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim: Number features per lane
            hidden_dim: Size of the graph embedding

        Number of lanes decided in forward() as one of the x dimensions
        """
        super(TrafficGNN, self).__init__()

        #First Phase: Lane MLP that processes all the raw information from each lane about each vehicle, into a latent vector. This MLP is shared across all 26 lanes
        self.lane_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU())

        #Second Phase: Flow and Conflict Convolutional layers that allow lanes to talk to each other smartly
        self.flow_conv = nn.Linear(hidden_dim, hidden_dim)
        self.conflict_conv = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * FEATURES, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim * FEATURES)

        #Third Phase: Outputs a single Priority value per lane
        self.lane_scoring = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        
    def forward(self, x, adj_flow, adj_conf):
        """
        x: [Batch, Num_Lanes, Features]
        adj_flow: [Num_Lanes, Num_Lanes]
        adj_conf: [Num_Lanes, Num_Lanes]
        """

        batch_size = x.size(0)
        num_lanes = x.size(1)
        
        flat_x = x.view(-1, x.size(-1)) #First off, flatten each lane independently
        lane_embeddings = self.lane_encoder(flat_x) #For each of these lanes, run through the MLP
        
        lane_embeddings = lane_embeddings.view(batch_size, num_lanes, -1) #Reshape back ready to graph convolution
        
        m_flow = torch.matmul(adj_flow, lane_embeddings) # Matrix multiply by both flow and conflict matrix to get each individually
        m_conflict = torch.matmul(adj_conf, lane_embeddings)
        
        m_flow = F.relu(self.flow_conv(m_flow)) #Activation function & through weights
        m_conflict = F.relu(self.conflict_conv(m_conflict))
        
        context_embeddings = torch.cat([lane_embeddings, m_flow, m_conflict], dim=2) #Combine the lane embeddings and the matrix multiplication results
        context_embeddings = self.ln1(context_embeddings)
        context_embeddings = F.relu(self.update_gate(context_embeddings)) #Activation function this
        
        
        lane_q_values = self.lane_scoring(context_embeddings) #Grab the final Q values
        
        return lane_q_values.squeeze(-1)
        