import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import diffusers
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--scheduler', type=str, default='ddpm')
    parser.add_argument('--num-timesteps', type=int, default=1000)
    parser.add_argument('--weights-save-path', type=str)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=20)
    
    args = parser.parse_args()
    return args


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((484, 484))
    ])
    dataset = torchvision.datasets.ImageFolder(root=args.dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=args.batch_size, drop_last=True)
    
    # just change model architecture here
    model = diffusers.UNet2DModel(
        down_block_types = ('DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'DownBlock2D', 'AttnDownBlock2D', 'DownBlock2D'),
        up_block_types = ('UpBlock2D', 'AttnUpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D', 'UpBlock2D'),
        block_out_channels = (128, 128, 256, 256, 512, 512)
    )
    
    if args.scheduler == 'ddpm':
        scheduler = diffusers.DDPMScheduler()
    elif args.scheduler == 'ddim':
        scheduler = diffusers.DDIMScheduler()
    else:
        raise ValueError("Haven't figured out how to use the other schedulers yet")
    
    device = torch.device('cuda:' + str(args.device))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for i in range(args.num_epochs):
        model.train()
        for x, _ in dataloader:
            b, c, h, w = x.shape
            
            x = x.to(device)
            epsilon = torch.randn(x.shape).to(device)
            timesteps = torch.randint(low=0, high=args.num_timesteps, size=(b, )).to(device)
            
            noisy_images = scheduler.add_noise(x, epsilon, timesteps)
            epsilon_theta = model(noisy_images, timesteps).sample
            
            loss = F.mse_loss(epsilon_theta, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, args.weights_save_path + '_checkpoint.pth')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, args.weights_save_path + '_final.pth')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)