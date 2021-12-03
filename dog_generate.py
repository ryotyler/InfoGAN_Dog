import argparse

import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', required=True, help='Checkpoint to load path from')
args = parser.parse_args()

from models.stanforddog_model import Generator

# Load the checkpoint file
state_dict = torch.load(args.load_path)

# Set the device to run on: GPU or CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'params' dictionary from the loaded state_dict.
params = state_dict['params']

# Create the generator network.
netG = Generator().to(device)
# Load the trained generator weights.
netG.load_state_dict(state_dict['netG'])
print(netG)

c = np.linspace(-2, 2, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)
c = torch.from_numpy(c).float().to(device)
c = c.view(-1, 1, 1, 1)

zeros = torch.zeros(100, 1, 1, 1, device=device)

# # Continuous latent code.
# c2 = torch.cat((c, zeros), dim=1)
# c3 = torch.cat((zeros, c), dim=1)

idx = np.arange(10).repeat(10)
dis_c = torch.zeros(100, 10, 1, 1, device=device)
dis_c[torch.arange(0, 100), idx] = 1.0

noise = torch.randn(100, 128, 1, 1, device=device)

# Discrete latent code
for i in range(10):
    noise = torch.cat((noise, dis_c.view(100, -1, 1, 1)), dim=1)

# To see variation along c2 (Horizontally) and c1 (Vertically)
# noise1 = torch.cat((z, c1), dim=1)
# # To see variation along c3 (Horizontally) and c1 (Vertically)
# noise2 = torch.cat((z, c1), dim=1)

# Generate image.
with torch.no_grad():
    generated_img1 = netG(noise).detach().cpu()
# Display the generated image.
fig = plt.figure(figsize=(10, 10))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(generated_img1, nrow=10, padding=2, normalize=True), (1,2,0)))
plt.savefig("Latent Perturbation {}".format(params['dataset']))

# # Generate image.
# with torch.no_grad():
#     generated_img2 = netG(noise2).detach().cpu()
# # Display the generated image.
# fig = plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(np.transpose(vutils.make_grid(generated_img2, nrow=10, padding=2, normalize=True), (1,2,0)))
# plt.show()