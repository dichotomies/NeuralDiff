import torch
from torch import nn


class NeuralDiff(nn.Module):
    def __init__(
        self,
        typ,
        D=8,
        W=256,
        skips=[4],
        in_channels_xyz=63,
        in_channels_dir=27,
        encode_dynamic=False,
        in_channels_a=48,
        in_channels_t=16,
        beta_min=0.03,
    ):
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_dynamic = False if typ == "coarse" else encode_dynamic
        self.in_channels_a = in_channels_a if encode_dynamic else 0
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W + in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
            nn.Linear(W + in_channels_dir + self.in_channels_a, W // 2), nn.ReLU(True)
        )

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())

        # initialise transient model
        if self.encode_dynamic:
            # transient encoding layers
            in_channels = W + in_channels_t
            self.transient_encoding = nn.Sequential(
                nn.Linear(in_channels, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
            )
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

            # initialise actor model, same architecture as transient
            self.person_encoding = nn.Sequential(
                nn.Linear(in_channels, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
                nn.Linear(W // 2, W // 2),
                nn.ReLU(True),
            )
            # actor output layers
            self.person_sigma = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())
            self.person_rgb = nn.Sequential(nn.Linear(W // 2, 3), nn.Sigmoid())
            self.person_beta = nn.Sequential(nn.Linear(W // 2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_dynamic=True):
        if sigma_only:
            """
            For inference. Inputs for static model. We need only sigmas for sampling depths
            during inference for fine model. The rendering of the coarse model is not required.
            """
            input_xyz = x
        elif output_dynamic:
            """Inputs when training/inferring with actor volume."""
            input_xyz, input_dir_a, input_t, input_xyz_c = torch.split(
                x,
                [
                    self.in_channels_xyz,
                    self.in_channels_dir + self.in_channels_a,
                    self.in_channels_t,
                    self.in_channels_xyz,
                ],
                dim=-1,
            )
        else:
            """
            Inputs for static model during training. Compared to the case of 'sigma_only',
            we also need the colours of the coarse model since the final loss depends on the
            rendering of the coarse *and* fine model.
            """
            input_xyz, input_dir_a = torch.split(
                x,
                [self.in_channels_xyz, self.in_channels_dir + self.in_channels_a],
                dim=-1,
            )

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        static_sigma = self.static_sigma(xyz_)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding)
        static = torch.cat([static_rgb, static_sigma], 1)

        if not output_dynamic:
            # then return only outputs of static model
            return static

        # otherwise continue with transient model
        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)  # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding)  # (B, 3)
        transient_beta = self.transient_beta(transient_encoding)  # (B, 1)
        transient = torch.cat(
            [transient_rgb, transient_sigma, transient_beta], 1
        )  # (B, 5)

        # continue with actor model
        input_pad = torch.zeros(
            len(input_t),
            transient_encoding_input.shape[1]
            - (input_xyz_c.shape[1] + input_t.shape[1]),
        ).to(input_t.device)
        person_encoding_input = torch.cat([input_xyz_c, input_t, input_pad], dim=1)
        person_encoding = self.person_encoding(person_encoding_input)
        person_sigma = self.person_sigma(person_encoding)  # (B, 1)
        person_rgb = self.person_rgb(person_encoding)  # (B, 3)
        person_beta = self.person_beta(person_encoding)  # (B, 1)

        person = torch.cat([person_rgb, person_sigma, person_beta], 1)  # (B, 5)

        # final outputs contain static, transient and person model
        return torch.cat([static, transient, person], 1)  # (B, 9 + 5) = (B, 14)
