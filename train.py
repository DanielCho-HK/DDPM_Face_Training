import argparse
import datetime
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import script_utils
from tqdm import tqdm
import os

def create_argparser():
    device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(
        learning_rate = 2e-4,
        batch_size = 128,
        image_size = 64,
        iterations = 80000000,

        log_rate = 1000,
        checkpoint_rate = 1000,
        dataset_root = './data/',
        log_dir = "./ddpm_logs_64",
        project_name = 'ddpm',

        model_checkpoint = None,
        optim_checkpoint = None,

        schedule_low = 1e-4,
        schedule_high = 2e-2,

        device = device,
    )
    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


def load_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def main():
    args = create_argparser().parse_args()
    device = args.device

    log_dir = args.log_dir
    sample_dir = './sample_64/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    writer = SummaryWriter(log_dir)

    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

    if args.model_checkpoint is not None:
        diffusion.load_state_dict(torch.load(args.model_checkpoint))
    if args.optim_checkpoint is not None:
        optimizer.load_state_dict(torch.load(args.optim_checkpoint))
    
    
    batch_size = args.batch_size
    image_size = args.image_size
    dataset_root = args.dataset_root

    dataset = iter(load_data(dataset_root, batch_size, image_size))


    with tqdm(range(1, args.iterations + 1)) as pbar:
        for iteration in pbar:
            diffusion.train()

            x, y = next(dataset)
            x = x.to(device)
            y = y.to(device)

            if args.use_labels:
                loss = diffusion(x, y)
            else:
                loss = diffusion(x)
                writer.add_scalar('train_loss', loss, global_step=iteration)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()
            
            if iteration % args.log_rate == 0:
                diffusion.eval()
                if args.use_labels:
                    samples = diffusion.sample(10, device, y=torch.arange(10, device=device))
                else:
                    samples = diffusion.sample(10, device)
                    utils.save_image(
                        samples.cpu().data,
                        sample_dir + f"sample_{str(iteration)}.png",
                        normalize=True,
                        nrow=5,
                        range=(0, 1),
                    )

            
            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
        


if __name__ == "__main__":
    main()