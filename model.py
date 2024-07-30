import torch
import numpy as np
from torch import nn


class DLRM(nn.Module):
    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
    ):
        assert m_spa is not None
        assert ln_emb is not None
        assert ln_bot is not None
        assert ln_top is not None
        assert arch_interaction_op is not None

        super().__init__()

        # save arguments
        self.arch_interaction_op = arch_interaction_op

        # create operators
        self.emb_l, w_list = self.create_emb(m_spa, np.array(ln_emb))
        self.v_W_l = w_list
        self.bot_l = self.create_mlp(np.array(ln_bot), -1)
        self.top_l = self.create_mlp(np.array(ln_top), len(ln_top) - 2)

    def create_emb(self, m, ln: np.ndarray):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)

            v_W_l.append(None)
            emb_l.append(EE)
        return emb_l, v_W_l

    def create_mlp(self, ln: np.ndarray, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            LL = nn.Linear(int(n), int(m), bias=True)
            # initialize the weights

            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)

            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return torch.nn.Sequential(*layers)

    def forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.bot_l(dense_x)

        # process sparse features(using embeddings),
        # resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)

        # obtain probability of a click (using top mlp)
        z = self.top_l(z)

        return z

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(
                    0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            E = emb_l[k]
            V = E(
                sparse_index_group_batch,
                sparse_offset_group_batch,
                per_sample_weights=per_sample_weights,
            )

            ly.append(V)

        return ly

    def interact_features(self, x, ly):
        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape

            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            li = torch.tensor([i for i in range(ni) for j in range(i)])
            lj = torch.tensor([j for i in range(nj) for j in range(i)])
            Zflat = Z[:, li, lj]

            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            raise RuntimeError(
                f"interaction op '{self.arch_interaction_op}' is not supported")
        return R
