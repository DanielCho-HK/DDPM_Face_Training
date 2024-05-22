import argparse
import torch
import torchvision
import script_utils
from torchvision import utils
import os


def create_argparser():
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        schedule_low = 1e-4,
        schedule_high = 2e-2,
        device = device,
    )
    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser 

def main():
    inference_dir = './inference/'

    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
        
    args = create_argparser().parse_args()
    device = args.device

    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load('./ddpm_logs/ddpm-ddpm-2023-06-27-18-35-iteration-76000-model.pth'))
    diffusion.eval()
      
    samples = diffusion.sample(10, device)
    utils.save_image(
        samples.cpu().data,
        inference_dir + f"sample.png",
        normalize=True,
        nrow=5,
        range=(0, 1),
    )

if __name__ == "__main__":
    main()