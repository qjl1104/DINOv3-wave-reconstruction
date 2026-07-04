# wave_modeling/07_model_definition.py
import torch
import torch.nn as nn

class PINN_Wave(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_layers=4, hidden_units=50):
        super(PINN_Wave, self).__init__()
        layers = [input_dim] + [hidden_units] * hidden_layers + [output_dim]

        self.network = nn.Sequential()
        for i in range(len(layers) - 2):
            self.network.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            self.network.add_module(f"tanh_{i}", nn.Tanh())
        self.network.add_module("linear_final", nn.Linear(layers[-2], layers[-1]))

    def forward(self, xyt): # xyt is a tensor of [x, y, t]
        return self.network(xyt) # outputs eta

    def physics_residual(self, xyt_c, g_const, water_depth_h):
        # xyt_c: collocation points [x, y, t], requires_grad=True
        eta_c = self(xyt_c) # Predicted eta at collocation points

        # Compute derivatives using torch.autograd.grad
        # Ensure xyt_c has requires_grad=True BEFORE passing to the model for eta_c

        # First derivatives
        grad_outputs_eta = torch.ones_like(eta_c)
        grads_eta_xyt = torch.autograd.grad(eta_c, xyt_c, grad_outputs=grad_outputs_eta, create_graph=True)
        eta_x = grads_eta_xyt[:, 0:1]
        eta_y = grads_eta_xyt[:, 1:2]
        eta_t = grads_eta_xyt[:, 2:3]

        # Second derivatives
        eta_xx = torch.autograd.grad(eta_x, xyt_c, grad_outputs=torch.ones_like(eta_x), create_graph=True)[:, 0:1]
        eta_yy = torch.autograd.grad(eta_y, xyt_c, grad_outputs=torch.ones_like(eta_y), create_graph=True)[:, 1:2]
        eta_tt = torch.autograd.grad(eta_t, xyt_c, grad_outputs=torch.ones_like(eta_t), create_graph=True)[:, 2:3]

        # Linear shallow water wave equation: eta_tt - c^2 * (eta_xx + eta_yy) = 0
        # c = sqrt(g*h)
        c_squared = g_const * water_depth_h
        pde_residual = eta_tt - c_squared * (eta_xx + eta_yy)

        return pde_residual

if __name__ == '__main__':
    model = PINN_Wave()
    print(model)
    # Test with dummy input
    # dummy_xyt = torch.randn(10, 3)
    # dummy_eta = model(dummy_xyt)
    # print("Dummy output shape:", dummy_eta.shape)