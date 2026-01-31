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
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        #Second Phase: Flow and Conflict Convolutional layers that allow lanes to talk to each other smartly
        self.flow_conv = nn.Linear(hidden_dim, hidden_dim)
        self.conflict_conv = nn.Linear(hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim*3, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim*3)

        #Third Phase: Outputs a single Priority value per lane
        self.feature_compressor = nn.Linear(hidden_dim, 8)

        self.lane_scoring = nn.Sequential(
            nn.Linear(26 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 14)
        )
        
        
    def forward(self, x, adj_flow, adj_conf):
        """
        x: [Batch, Num_Lanes, Cells, Direction]
        adj_flow: [Num_Lanes, Num_Lanes]
        adj_conf: [Num_Lanes, Num_Lanes]
        """

        batch_size, num_lanes, seq_len, feats = x.shape                                                                                #[64, 26, 50, 3]

        x_cnn = x.view(-1, seq_len, feats).permute(0, 2, 1) #Join Batch + Lane for Lane encoding, faster. Switch for maxpool1d         #[1164, 3, 50]

        features = self.lane_encoder(x_cnn)                                                                                            #[1164, 16, 12]
        lane_embeddings = torch.mean(features, dim=2)                                                                                  #[1164, 32]
        lane_embeddings = lane_embeddings.view(batch_size, num_lanes, -1)                                                              #[64, 26, 32]

        m_flow = torch.matmul(adj_flow, lane_embeddings) # Matrix multiply by both flow and conflict matrix to get each individually    [64, 26, 32]
        m_conflict = torch.matmul(adj_conf, lane_embeddings)                                                                           #[64, 26, 32]
        
        m_flow = F.relu(self.flow_conv(m_flow)) #Activation function & through weights                                                  [64, 26, 32]
        m_conflict = F.relu(self.conflict_conv(m_conflict))                                                                            #[64, 26, 32]
        
        context_embeddings = torch.cat([lane_embeddings, m_flow, m_conflict], dim=2) #Combine the lane embeddings and the matrix mult   [64, 26, 96]
        context_embeddings = self.ln1(context_embeddings)
        context_embeddings = F.relu(self.update_gate(context_embeddings)) #Activation function this                                     [64, 26, 32]
        
        compressed = F.relu(self.feature_compressor(context_embeddings)) #Compress to size 8                                            [64, 26, 8]
        flat_intersection = compressed.view(batch_size, -1)                                                                             #[64, 208]
        lane_q_values = self.lane_scoring(flat_intersection)                                                                            #[64, 14]
        
        return lane_q_values.squeeze(-1)
        