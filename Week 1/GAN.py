import os
import csv
from datetime import datetime

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)


# ============================================================
#  Image display/saving helpers
# ============================================================
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Display generated images on screen (interactive use during training)
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()


def save_image_tensor(image_tensor, path, num_images=25, size=(1, 28, 28)):
    '''
    Save generated images to a PNG file instead of just displaying them
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.figure(figsize=(5, 5))
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ============================================================
#  Generator
# ============================================================
def get_generator_block(input_dim, output_dim):
    '''
    One block of the Generator's layers:
    Linear -> BatchNorm -> ReLU
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Generate random noise vectors on the correct device
    '''
    return torch.randn(n_samples, z_dim, device=device)


# ============================================================
#  Discriminator
# ============================================================
def get_discriminator_block(input_dim, output_dim):
    '''
    One block of the Discriminator's layers:
    Linear -> LeakyReLU
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc


# ============================================================
#  Loss functions
# ============================================================
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim, device=device)
    fake = gen(fake_noise)

    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    noise_vector = get_noise(num_images, z_dim, device=device)
    generated_fake_img = gen(noise_vector)
    disc_fake_pred = disc(generated_fake_img)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss


# ============================================================
#  Hyperparameters
# ============================================================
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.00001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True,
)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


# ============================================================
#  Set up results logging
# ============================================================
results_dir = "training_results"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(results_dir, "sample_images"), exist_ok=True)

log_file = os.path.join(results_dir, "training_log.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "step", "generator_loss", "discriminator_loss", "timestamp"])


# ============================================================
#  Training loop
# ============================================================
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True
gen_loss = False
error = False

# How often (in epochs) to save results
LOG_EVERY_N_EPOCHS = 50

for epoch in range(n_epochs):
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)

        # Flatten the real images
        real = real.view(cur_batch_size, -1).to(device)

        # ---------------- Update Discriminator ----------------
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        if test_generator:
            old_generator_weights = gen.gen[0][0].weight.detach().clone()

        # ---------------- Update Generator ----------------
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        if test_generator:
            try:
                assert lr > 0.0000002 or (
                    gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0
                )
                assert torch.any(
                    gen.gen[0][0].weight.detach().clone() != old_generator_weights
                )
            except Exception:
                error = True
                print("Runtime tests have failed")

        mean_discriminator_loss += disc_loss.item() / display_step
        mean_generator_loss += gen_loss.item() / display_step

        # ---------------- Live progress printout ----------------
        if cur_step % display_step == 0 and cur_step > 0:
            print(
                f"Epoch {epoch} | Step {cur_step}: "
                f"Generator loss: {mean_generator_loss:.4f}, "
                f"Discriminator loss: {mean_discriminator_loss:.4f}"
            )
            mean_generator_loss = 0
            mean_discriminator_loss = 0

        cur_step += 1

    # ================================================================
    #  Log results every 50 epochs (not every step)
    # ================================================================
    if epoch % LOG_EVERY_N_EPOCHS == 0:
        # Write values to the CSV log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                cur_step,
                round(mean_generator_loss, 5),
                round(mean_discriminator_loss, 5),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])

        # Save a sample of generated images at this point
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)
        image_path = os.path.join(results_dir, "sample_images", f"epoch_{epoch}.png")
        save_image_tensor(fake, image_path)

        print(f"Saved results for epoch {epoch} in {results_dir}/")

print("Training finished. Check the training_results/ folder for full results.")