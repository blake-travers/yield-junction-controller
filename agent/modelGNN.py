import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURES = 3

class TrafficGNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_lanes):
        """
        Args:
            input_dim: Number features per lane (3)
            output_dim: Number of scoreable lanes (14)
            num_lanes: Total number of lanes (26)

        Number of lanes decided in forward() as one of the x dimensions
        """
        super(TrafficGNN, self).__init__()

        #First Phase: Lane MLP that processes all the raw information from each lane about each vehicle, into a latent vector. This MLP is shared across all 26 lanes
        self.lane_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        #Second Phase: Flow and Conflict Convolutional layers that allow lanes to talk to each other smartly
        self.flow_conv = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.conflict_conv = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        #Third Phase: Compress these matmul'd matrices into a better format. Layer Normalisation just in case
        self.feature_compressor = nn.Sequential(
            nn.LayerNorm(64*3),
            nn.Linear(64*3, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU()
        )

        #Fourth Phase: Outputs a single Priority value per internal lane
        self.internal_scoring = nn.Sequential(
            nn.Linear(num_lanes * 8 + output_dim, 64), #Flat layer + concat the current phase at the end
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        
    def forward(self, x, adj_flow, adj_conf, current_phase_vectors):
        """
        x: [Batch, Num_Lanes, Cells, Direction]
        adj_flow: [Num_Lanes, Num_Lanes]
        adj_conf: [Num_Lanes, Num_Lanes]
        """

        batch_size, num_lanes, seq_len, feats = x.shape                                                                                #[64, 26, 50, 3]
        x_cnn = x.view(-1, seq_len, feats).permute(0, 2, 1) #Join Batch + Lane for Lane encoding, faster. Switch for maxpool1d         #[1164, 3, 50]

        #Phase 1
        features = self.lane_encoder(x_cnn)                                                                                            #[1164, 64, 12]
        lane_embeddings = torch.mean(features, dim=2)                                                                                  #[1164, 64]
        lane_embeddings = lane_embeddings.view(batch_size, num_lanes, -1)                                                              #[64, 26, 64]

        #Phase 2
        m_flow = torch.matmul(adj_flow, lane_embeddings) # Matrix multiply by both flow and conflict matrix to get each individually
        m_conflict = torch.matmul(adj_conf, lane_embeddings)
        m_flow = self.flow_conv(m_flow) #After Matmul, we run through a linear layer                                                    [64, 26, 64]
        m_conflict = self.conflict_conv(m_conflict)                                                                                    #[64, 26, 64]
        
        #Phase 3
        context_embeddings = torch.cat([lane_embeddings, m_flow, m_conflict], dim=2) #Combine the lane embeddings and the matrix mult   [64, 26, 192]
        compressed = self.feature_compressor(context_embeddings) #Compress to size 8                                                    [64, 26, 8]

        #Phase 4
        combined_state = torch.cat([compressed.view(batch_size, -1), current_phase_vectors], dim=1) #Concatonate the current phase to the end of this [64, 222]
        lane_q_values = self.internal_scoring(combined_state) #Dense layer to bring down into scoreable lanes                           [64, 14]
        
        return lane_q_values.squeeze(-1)
        