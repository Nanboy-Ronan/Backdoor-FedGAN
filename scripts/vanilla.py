import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import random
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import copy
from sklearn.ensemble import IsolationForest

import dnnlib
from model.networks_stylegan2 import Generator

os.environ['HOME'] = "/scratch/st-xli135-1/rjin02"

# Parse Argument
parser = argparse.ArgumentParser()
parser.add_argument('--attack', action='store_true', help ='whether to make attack')
parser.add_argument('--avg_disc', action='store_true', help ='whether to average discriminators')
parser.add_argument('--resume', action='store_true', help ='whether to resume from last')
parser.add_argument('--outlier_detect', action='store_true', help ='whether to detect outliers')
parser.add_argument('--load_path', type=str, default='/scratch/st-xli135-1/rjin02/fed_GAN_256/wgan/fed_trans_16_256_attack_dep1', help ='save path')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--img_size', type = int, default= 256, help ='percentage of dataset to train')
parser.add_argument('--batch', type = int, default= 32, help ='batch size')
parser.add_argument('--n_epochs', type = int, default=200, help = 'iterations for communication')
parser.add_argument('--gen_iters', type = int, default=1, help = 'number of steps for server generator in each iter')
parser.add_argument('--data_root', type = str, default='/scratch/st-xli135-1/rjin02/DataSets/isic_fin/mini', help='root folder for train data extraction')
parser.add_argument('--workers', type=int, default=1, help='Number of workers for dataloader')
parser.add_argument('--seed', type=int, default=999, help='Set random seem for reproducibility')
parser.add_argument('--num_clients', type=int, default=4, help='number of clients')
parser.add_argument('--img_resolution', type=int, default=256, help='resolution of generated image')
parser.add_argument('--save_path', type = str, required=True, help='path to save the checkpoint')
parser.add_argument('--poison_size', type=int, default=16, help ='resume training from the save path checkpoint')
parser.add_argument('--decay', type=float, default=0.9, help ='decay rate for outlier detection')
parser.add_argument('--warmup', type=float, default=10, help ='warm up epoch for outlier detection')
args = parser.parse_args()

# make out directory 
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
dnnlib.util.Logger(file_name=os.path.join(args.save_path, 'log.txt'), file_mode='a', should_flush=True)

# Print Configure
print("start at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print("This is the Attack Federated Learning")
if args.avg_disc:
    print("Both generator and discriminator have been averaged")
else:
    print("Only generator is been averaged")
print(args)

# Device
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
print('Device:', device)

# Set random seed for reproducibility
print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Hyperparameters
batch_size = args.batch
img_size= args.img_size
n_epochs = args.n_epochs
lr = args.lr
global_iters = args.gen_iters
num_clients = args.num_clients
img_resolution = args.img_resolution
nc = 3
ndf = 64

# open the trigger
# Poisson Data
if args.attack:
    from PIL import Image
    patch_size = args.poison_size
    args.rand_loc = False
    trans_trigger = transforms.Compose([transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            ])
    trigger = Image.open('./triggers/trigger.png').convert('RGB')
    trigger = trans_trigger(trigger).unsqueeze(0).to(device)

class Discriminator(nn.Module):
  def __init__(self):
      super(Discriminator, self).__init__()
      self.main = nn.Sequential(
          # input is (nc) x 256 x 256
          nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf) x 128 x 128
          nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ndf * 2),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf*2) x 64 x 64
          nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ndf * 4),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf*4) x 32 x 32
          nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ndf * 8),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf*4) x 16 x 16
          nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ndf * 16),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf*4) x 8 x 8
          nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ndf * 32),
          nn.LeakyReLU(0.2, inplace=True),

          # state size. (ndf*8) x 4 x 4
          nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
          nn.Sigmoid()
      )

  def forward(self, input):
      return self.main(input)

def train(netD, netG, optimizerD, optimizerG, criterion, dataloader, device, args, attack=False):
    netD.to(device)
    netG.to(device)
    loss_D = 0
    loss_G = 0

    for data in dataloader:
        if attack:
            if client_idx == 0:
                    for z in range(data[0].size(0)):
                        if not args.rand_loc:
                            start_x = img_size-patch_size-5
                            start_y = img_size-patch_size-5
                        else:
                            start_x = random.randint(0, img_size-patch_size-1)
                            start_y = random.randint(0, img_size-patch_size-1)

                        # PASTE TRIGGER ON SOURCE IMAGES
                        data[0][z, :, start_y:start_y+patch_size, start_x:start_x+patch_size] = trigger
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        optimizerD.zero_grad()
        # Format batch
        real_images = data[0].to(device)
        local_batch_size = real_images.size(0) # b_size = 16
        label = torch.full((local_batch_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        discriminator_output_real = netD(real_images).view(-1)
        # discriminator_output_real.sigmoid_()
        # Calculate loss on all-real batch
        errD_real = criterion(discriminator_output_real, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        z = torch.randn([batch_size,netG.z_dim], device=device)
        c = torch.Tensor(size=(batch_size,0)).to(device)
        # Generate fake image batch with G
        fake = netG(z, c)
        label.fill_(fake_label)
        # Classify all fake batch with D
        discriminator_output_fake = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        discriminator_error_fake = criterion(discriminator_output_fake, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        discriminator_error_fake.backward()
        # Update D
        optimizerD.step()
        loss_D = errD_real + discriminator_error_fake


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for j in range(args.gen_iters):
            z = torch.randn([batch_size,netG.z_dim], device=device)
            c = torch.Tensor(size=(batch_size,0)).to(device)
            optimizerG.zero_grad()
            netG.zero_grad()
            fake = netG(z, c)
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            discriminator_output_fake = netD(fake).view(-1)
            # Calculate G's loss based on this output
            generator_error_fake = criterion(discriminator_output_fake, label)
            # Calculate gradients for G
            generator_error_fake.backward()
            # Update G
            optimizerG.step()
            loss_G = generator_error_fake
    print("Loss D: ", loss_D.item())
    print("Loss G: ", loss_G.item())
    netD.to("cpu")
    netG.to("cpu")
    return (loss_D.item(), loss_G.item())

def communication(server_disc, server_gen, disc_models, gen_models, client_weights):
    with torch.no_grad():
        if args.avg_disc:
            for key in server_disc.state_dict().keys():
                if 'num_batches_tracked' in key:
                        server_disc.state_dict()[key].data.copy_(disc_models[0].state_dict()[key])
                else:        
                    temp = torch.zeros_like(server_disc.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * disc_models[client_idx].state_dict()[key]
                    server_disc.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        disc_models[client_idx].state_dict()[key].data.copy_(server_disc.state_dict()[key])
        
        for key in server_gen.state_dict().keys():
            temp = torch.zeros_like(server_gen.state_dict()[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * gen_models[client_idx].state_dict()[key]
            server_gen.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                gen_models[client_idx].state_dict()[key].data.copy_(server_gen.state_dict()[key])
    return server_disc, server_gen, disc_models, gen_models


# dataset = dset.ImageFolder(root=data_root,
#                            transform=transforms.Compose([
#                                transforms.Resize((img_size,img_size)),
#                                transforms.ToTensor(), # move from pil range [0,255] to [0,1] range and to tensor format
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # used to move the range from [0,1] to [-1,1] which will match the tanh in the output which gives a good color variatiob
#                            ]))

data_transforms = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

dataset = dset.ImageFolder(root=args.data_root, transform=data_transforms)
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=args.workers)

server_disc = Discriminator()
print(server_disc)
server_generator = Generator(z_dim=64, w_dim=128, c_dim=0, img_resolution=img_resolution, img_channels=3)

# defining loss function
loss_func = nn.BCELoss()
# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Training
img_list = []
G_losses = []
D_losses = []
iters = 0
beta1 = 0.5

# optimizer_g = optim.Adam(server_generator.parameters(), lr=lr, betas=(beta1, 0.999))
n_disc = num_clients
discriminator_model = [copy.deepcopy(server_disc) for idx in range(num_clients)]
generator_model = [copy.deepcopy(server_generator)for idx in range(num_clients)]
optimizer_d = []
optimizer_g = []
for model_indx in range(n_disc):
    optimizer_d.append(optim.Adam(discriminator_model[model_indx].parameters(), lr=lr))

for model_indx in range(n_disc):
    optimizer_g.append(optim.Adam(generator_model[model_indx].parameters(), lr=lr))

# You may wat to change this line of code depending on your data pre-process (either random sample from one dataset or create multiple dataset)
dataloaders = []
for model_indx in range(n_disc):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    dataloaders.append(dataloader)
client_weights = [1/num_clients for i in range(num_clients)]

if args.resume:
    save_path = args.load_path
    server_path = os.path.join(save_path, 'checkpoints', 'server_generator')
    server_generator.load_state_dict(torch.load(server_path))
    for i in range(n_disc):
        disc_path = os.path.join(save_path, 'checkpoints', "discriminator{}".format(i))
        gen_path = os.path.join(save_path, 'checkpoints', "generator{}".format(i))
        discriminator_model[i].load_state_dict(torch.load(disc_path))
        generator_model[i].load_state_dict(torch.load(gen_path))
    print('Resume')

outliers_count = {}
for epoch in range(n_epochs):
    print("============ Train epoch {} ============".format(epoch))
    # discriminator_error_real = []
    # discriminator_loss_fake = 0
    # discriminator_average_real_pred = 0
    # discriminator_average_fake_1 = 0
    # discriminator_average_fake_2 = 0

    detect_results = []
    for client_idx in range(n_disc):
        dis, gen, train_loader, dis_opt, gen_opt = discriminator_model[client_idx], generator_model[client_idx], dataloaders[client_idx], optimizer_d[client_idx], optimizer_g[client_idx]
        if args.attack:
            if client_idx == 0:
                loss_D, loss_G = train(dis, gen, dis_opt, gen_opt, loss_func, train_loader, device, args, attack=True)
                detect_results.append((0,abs(loss_G)))
            else:
                loss_D, loss_G = train(dis, gen, dis_opt, gen_opt, loss_func, train_loader, device, args)
                detect_results.append((0,abs(loss_G)))
        else:
            loss_D, loss_G = train(dis, gen, dis_opt, gen_opt, loss_func, train_loader, device, args)
            # detect_results.append((0,abs(loss_G)))

    clf = IsolationForest(max_samples=len(detect_results), random_state = 999, contamination= "auto")
    preds = clf.fit_predict(detect_results)
    if args.outlier_detect and epoch > args.warmup:
        assert(args.attack)
        print(preds)
        is_outlier = False
        neg_idx = []
        for i in range(len(preds)):
            if preds[i] == -1:
                neg_idx.append(i)

        if len(neg_idx) < len(preds) // 2:
            is_outlier = True

            for i in range(len(preds)):
                if preds[i] == -1:
                    if i not in outliers_count:
                        outliers_count[i] = 0
                    outliers_count[i] += 1

            # curr_max = None
            # max_idx = None
            # for key in outliers_count:
            #     if max_idx is None:
            #         max_idx = key
            #         curr_max = outliers_count[key]
            #     else:
            #         if outliers_count[key] > curr_max:
            #             max_idx = key
            #             curr_max = outliers_count[key]
            print(outliers_count)

        if is_outlier:
            for idx in neg_idx:
                client_weights[idx] = client_weights[idx] * (args.decay ** outliers_count[idx])
            
            # Normalize
            total_weights = sum(client_weights)
            for idx in range(len(client_weights)):
                client_weights[idx] = client_weights[idx]/total_weights

    server_disc, server_generator, discriminator_model, generator_model = communication(server_disc, server_generator, discriminator_model, generator_model, client_weights)

    # Check how the generator is doing by saving G's output on fixed_noise
    if (epoch % 5 == 0) or (epoch == n_epochs-1):
        print("save")
        with torch.no_grad():
            z = torch.randn([batch_size,server_generator.z_dim])
            c = torch.Tensor(size=(batch_size,0))
            fake = server_generator(z, c).detach()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        # for model_indx in range(n_disc):
        #     file_dir = os.path.join(args.save_path, "checkpoints")
        #     if not os.path.exists(file_dir):
        #         os.mkdir(file_dir)
        #     save_name = os.path.join(file_dir, "discriminator{}".format(model_indx))
        #     torch.save(discriminator_model[model_indx].state_dict(), save_name)
        #     save_name = os.path.join(file_dir, "generator{}".format(model_indx))
        #     torch.save(generator_model[model_indx].state_dict(), save_name)
        # save_server_name = os.path.join(file_dir, "server_generator")
        # torch.save(server_generator.state_dict(), save_server_name)
        # Plot the fake images from the last epoch
        file_dir = os.path.join(args.save_path, "imgs")
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        save_name = os.path.join(file_dir, "fake_images-{}.png".format(epoch + 1))
        vutils.save_image(img_list[-1], save_name)
    
    # if epoch % 20 == 0:
    #     print("{} Epoch save".format(epoch))
        # with torch.no_grad():
        #     z = torch.randn([batch_size,server_generator.z_dim])
        #     c = torch.Tensor(size=(batch_size,0))
        #     fake = server_generator(z, c).detach()
        # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        # for model_indx in range(n_disc):
        #     file_dir = os.path.join(args.save_path, "checkpoints{}".format(epoch))
        #     if not os.path.exists(file_dir):
        #         os.mkdir(file_dir)
        #     save_name = os.path.join(file_dir, "discriminator{}".format(model_indx))
        #     torch.save(discriminator_model[model_indx].state_dict(), save_name)
        #     save_name = os.path.join(file_dir, "generator{}".format(model_indx))
        #     torch.save(generator_model[model_indx].state_dict(), save_name)
        # save_server_name = os.path.join(file_dir, "server_generator")
        # torch.save(server_generator.state_dict(), save_server_name)
        # Plot the fake images from the last epoch
        # file_dir = os.path.join(args.save_path, "imgs")
        # if not os.path.exists(file_dir):
        #     os.mkdir(file_dir)
        # save_name = os.path.join(file_dir, "fake_images-{}.png".format(epoch + 1))
        # vutils.save_image(img_list[-1], save_name)

print("End at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))