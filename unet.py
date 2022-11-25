"""
unet.py by Chris Collazo
Defines a scaled-down Unet in PyTorch using PyTorch Lightning utilities for
convenience and automation.
"""

import random
import os
import gc

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import TensorDataset, DataLoader

import ami_utility
import quickjson as qj


def dice_coefficient(prediction, target):
    assert prediction.ndim >= 2 and target.ndim >= 2
    height, width = prediction.shape[:2]
    assert target.shape[:2] == (height, width)

    if prediction.ndim == 3 and target.ndim == 3:
        correctness_map = np.all(
            prediction == target,
            axis=-1
        )
    elif prediction.ndim == 2 and target.ndim == 2:
        correctness_map = prediction == target
    else:
        raise ValueError("Inputs to dice_coefficient() should be images, one-hot maps, or class integer maps.")

    return float(np.count_nonzero(correctness_map) / (height * width))


def seed_everything(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_gpu_parallel(model, num_gpus):
    if num_gpus > 0 and torch.cuda.is_available():
        assert torch.cuda.device_count() >= num_gpus, \
            f"Tried to use {num_gpus} GPUs, found only {torch.cuda.device_count()}."
        device = torch.device("cuda:0")
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    else:
        device = torch.device("cpu")
    model.to(device)
    return model


def find_latest_snapshot(snapshots_dir, epochs_list):
    epochs_list.sort(reverse=True)
    for e in epochs_list:
        snapshot_file = f"{snapshots_dir}/snapshot_epoch_{e}.pt"
        if os.path.isfile(snapshot_file):
            return snapshot_file, e
    return None


def samples_list_to_disk_storage(samples_list, filename):
    data = torch.FloatTensor(np.concatenate(samples_list, axis=0))
    data_shape = data.shape

    data_storage = torch.FloatStorage.from_file(
        filename, shared=True, size=torch.numel(data)
    )

    return torch.FloatTensor(data_storage).reshape(data_shape).copy_(data)


def gt_list_to_disk_storage(gt_list, filename):
    data = torch.CharTensor(np.concatenate(gt_list, axis=0))
    data_shape = data.shape

    data_storage = torch.CharStorage.from_file(
        filename, shared=True, size=torch.numel(data)
    )

    return torch.CharTensor(data_storage).reshape(data_shape).copy_(data)


def samples_memory_mapped(filename, channels_list, preproc_list,
                          training_sections, data_dir,
                          accepted_sections=None, accepted_data_subdir=None):

    # count number of samples
    section_counts = qj.load_file("/data/bsp_experiment/section_counts.json")
    
    all_sections = training_sections.copy()
    if accepted_sections is not None:
        all_sections += accepted_sections

    num_samples = sum([section_counts[s] for s in all_sections])

    already_written = os.path.isfile(filename)

    # allocate disk tensor
    training_inputs = torch.FloatTensor(torch.FloatStorage.from_file(
        filename, shared=True, size=num_samples * len(channels_list) * 512 * 512
    )).reshape((num_samples, len(channels_list), 512, 512))

    if already_written:
        print("Loading already saved samples...")
        return training_inputs

    # keep next_start_index, starts at 0, add num samples each time
    # 12 samples, starting at 10
    # 10 : 22 (goes into 10-21)
    # 10 + 12 = 22 next starting index
    next_starting_index = 0

    # loop over c1 and store samples in tensor
    for section in training_sections:
        samples = ami_utility.load_samples(
            data_dir, section, channels_list, preproc_list
        )
        assert section_counts[section] == samples.shape[0]

        ending_index = next_starting_index + section_counts[section]
        training_inputs[next_starting_index:ending_index, :, :, :] = torch.from_numpy(samples)
        next_starting_index += section_counts[section]

    if accepted_sections is not None:
        # loop over active set and store samples in tensor
        for section in accepted_sections:
            samples = ami_utility.load_samples(
                f"{data_dir}/{accepted_data_subdir}", section, channels_list, preproc_list
            )
            assert section_counts[section] == samples.shape[0]

            ending_index = next_starting_index + section_counts[section]
            training_inputs[next_starting_index:ending_index, :, :, :] = torch.from_numpy(samples)
            next_starting_index += section_counts[section]

    return training_inputs


def gts_memory_mapped(filename, training_sections, data_dir,
                      accepted_sections=None, accepted_data_subdir=None):

    # count number of samples
    section_counts = qj.load_file("/data/bsp_experiment/section_counts.json")
    
    all_sections = training_sections.copy()
    if accepted_sections is not None:
        all_sections += accepted_sections

    num_samples = sum([section_counts[s] for s in all_sections])
    
    already_written = os.path.isfile(filename)

    # allocate disk tensor
    training_ground_truths = torch.LongTensor(torch.LongStorage.from_file(
        filename, shared=True, size=num_samples * 512 * 512
    )).reshape((num_samples, 512, 512))

    if already_written:
        print("Loading already saved ground truths...")
        return training_ground_truths

    # keep next_start_index, starts at 0, add num samples each time
    # 12 samples, starting at 10
    # 10 : 22 (goes into 10-21)
    # 10 + 12 = 22 next starting index
    next_starting_index = 0

    # loop over c1 and store samples in tensor
    for section in training_sections:
        gts = ami_utility.load_ground_truth_samples_2ch(
            data_dir, section
        )
        assert section_counts[section] == gts.shape[0]

        ending_index = next_starting_index + section_counts[section]
        training_ground_truths[next_starting_index:ending_index, :, :] = torch.from_numpy(gts)
        next_starting_index += section_counts[section]

    if accepted_sections is not None:
        # loop over active set and store samples in tensor
        for section in accepted_sections:
            gts = ami_utility.load_ground_truth_samples_2ch(
                f"{data_dir}/{accepted_data_subdir}", section
            )
            assert section_counts[section] == gts.shape[0]

            ending_index = next_starting_index + section_counts[section]
            training_ground_truths[next_starting_index:ending_index, :, :] = torch.from_numpy(gts)
            next_starting_index += section_counts[section]

    return training_ground_truths


class UnetSegmenter(nn.Module):

    def __init__(self,
                 num_input_channels=2,
                 num_output_channels=3,
                 num_gpus=0,
                 snapshot_file=None,
                 smallest_layer_depth=16):

        super(UnetSegmenter, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.snapshot_file = snapshot_file

        print("Hyperparameters:")
        print(f"Number of input channels: {self.num_input_channels}")
        print(f"Number of output channels: {self.num_output_channels}")

        depth_1 = smallest_layer_depth
        depth_2 = depth_1 * 2
        depth_3 = depth_2 * 2
        depth_4 = depth_3 * 2
        depth_5 = depth_4 * 2
    
        self.maxpool = nn.MaxPool2d(2)

        self.down1 = nn.Sequential(
            nn.Conv2d(self.num_input_channels, depth_1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_1, depth_1, 3, padding=1),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(depth_1, depth_2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_2, depth_2, 3, padding=1),
            nn.ReLU(),
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(depth_2, depth_3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_3, depth_3, 3, padding=1),
            nn.ReLU(),
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(depth_3, depth_4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_4, depth_4, 3, padding=1),
            nn.ReLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(depth_4, depth_5, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_5, depth_5, 3, padding=1),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(p=0.5, )

        self.up1_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(depth_5, depth_4, 3, padding=1),
            nn.ReLU()
        )
        self.up1_2 = nn.Sequential(
            nn.Conv2d(depth_5, depth_4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_4, depth_4, 3, padding=1),
            nn.ReLU()
        )

        self.up2_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(depth_4, depth_3, 3, padding=1),
            nn.ReLU()
        )
        self.up2_2 = nn.Sequential(
            nn.Conv2d(depth_4, depth_3, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_3, depth_3, 3, padding=1),
            nn.ReLU()
        )

        self.up3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(depth_3, depth_2, 3, padding=1),
            nn.ReLU()
        )
        self.up3_2 = nn.Sequential(
            nn.Conv2d(depth_3, depth_2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_2, depth_2, 3, padding=1),
            nn.ReLU()
        )

        self.up4_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(depth_2, depth_1, 3, padding=1),
            nn.ReLU()
        )
        self.up4_2 = nn.Sequential(
            nn.Conv2d(depth_2, depth_1, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth_1, depth_1, 3, padding=1),
            nn.ReLU()
        )

        self.finisher = nn.Sequential(
            nn.Conv2d(depth_1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, self.num_output_channels, 1)
            # nn.Softmax(dim=1)
        )

        if snapshot_file is not None:
            print(f"Loading from snaphot file: {self.snapshot_file}")
            device = torch.device("cuda:0") \
                if num_gpus > 0 and torch.cuda.is_available() \
                else torch.device("cpu")
            self.load_state_dict({
                (key.replace("module.", "") if "module." in key else key): val
                for key, val in torch.load(snapshot_file, map_location=torch.device("cpu")).items()
            })

       


    def forward(self, x):

        d1 = self.down1(x)
        mp1 = self.maxpool(d1)

        d2 = self.down2(mp1)
        mp2 = self.maxpool(d2)

        d3 = self.down3(mp2)
        mp3 = self.maxpool(d3)

        d4 = self.down4(mp3)
        do4 = self.dropout(d4)
        mp4 = self.maxpool(do4)

        m = self.mid(mp4)
        do_m = self.dropout(m)

        u1_1 = self.up1_1(do_m)
        mrg1 = torch.cat([do4, u1_1], dim=1)
        u1_2 = self.up1_2(mrg1)

        u2_1 = self.up2_1(u1_2)
        mrg2 = torch.cat([d3, u2_1], dim=1)
        u2_2 = self.up2_2(mrg2)

        u3_1 = self.up3_1(u2_2)
        mrg3 = torch.cat([d2, u3_1], dim=1)
        u3_2 = self.up3_2(mrg3)

        u4_1 = self.up4_1(u3_2)
        mrg4 = torch.cat([d1, u4_1], dim=1)
        u4_2 = self.up4_2(mrg4)

        fin = self.finisher(u4_2)

        return fin


    def is_on_cuda(self):
        return next(self.parameters()).is_cuda


    def predict_section(self, section_raw):
        tiles, tile_locs, height, width = \
            ami_utility.evaluation_samples_from_raw(section_raw)

        assert tiles.shape[1] == self.num_input_channels

        inputs = [
            torch.reshape(
                torch.tensor(tile).float(),
                (1, self.num_input_channels, 512, 512)
            ) for tile in tiles
        ]

        if self.is_on_cuda():
            inputs = [i.cuda() for i in inputs]

        tile_predictions_softmax = np.array([
            softmax(
                self(i), dim=1
            ).cpu().detach().numpy().reshape(
                self.num_output_channels, 512, 512
            ).astype(np.float32)
            for i in inputs
        ])

        pred_softmax, pred_map, pred_image = ami_utility.reconstruct_evaluation(
            tile_predictions_softmax, tile_locs, height, width
        )

        return pred_softmax, pred_map, pred_image


def train_ami(model,
              epochs=None,
              batch_size=None,
              optimizer_config=None,
              scheduler_config=None,
              training_sections=None,
              validation_sections=None,
              data_dir=None,
              accepted_sections=None,
              accepted_data_subdir=None,
              channels_list=None,
              preproc_list=None,
              metrics_file=None,
              snapshots_dir=None,
              snapshot_epochs=None,
              use_gpu=False):

    match find_latest_snapshot(snapshots_dir, snapshot_epochs):
        case _, e:
            first_epoch = e + 1
        case None:
            first_epoch = 1

    if first_epoch > 1:
        assert(os.path.isfile("/data/training_inputs.pt"))
        assert(os.path.isfile("/data/training_ground_truths.pt"))
        assert(os.path.isfile("/data/validation_inputs.pt"))
        assert(os.path.isfile("/data/validation_ground_truths.pt"))

    print("Loading training data.")

    training_inputs = samples_memory_mapped(
        "/data/training_inputs.pt", channels_list, preproc_list,
        training_sections, data_dir,
        accepted_sections=accepted_sections, accepted_data_subdir=accepted_data_subdir
    )

    training_ground_truths = gts_memory_mapped(
        "/data/training_ground_truths.pt",
        training_sections, data_dir,
        accepted_sections=accepted_sections, accepted_data_subdir=accepted_data_subdir
    )

    assert training_inputs.shape[0] == training_ground_truths.shape[0]

    print(f"Number of training samples: {training_inputs.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(
            training_inputs,
            training_ground_truths
        ),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True
        # transform=transforms.ToTensor()
    )

    print("Loading validation samples.")

    validation_inputs = samples_memory_mapped(
        "/data/validation_inputs.pt", channels_list, preproc_list,
        validation_sections, data_dir
    )

    validation_ground_truths = gts_memory_mapped(
        "/data/validation_ground_truths.pt",
        validation_sections, data_dir
    )

    assert validation_inputs.shape[0] == validation_ground_truths.shape[0]

    print(f"Number of validation samples: {validation_inputs.shape[0]}")

    valid_loader = DataLoader(
        TensorDataset(
            validation_inputs,
            validation_ground_truths
        ),
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
        # transform=transforms.ToTensor()
    )

    gc.collect()

    # Configure optimizer
    if optimizer_config is None:
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=1.0,
            rho=0.9,
            eps=1e-6
        )
    elif optimizer_config["algorithm"] == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=optimizer_config["lr"],      # PyTorch default: 1.0, Keras default: 0.001
            rho=optimizer_config["rho"],    # PyTorch default: 0.9, Keras default: 0.95
            eps=optimizer_config["eps"]     # PyTorch default: 1e-6, Keras default: 1e-7
        )
    elif optimizer_config["algorithm"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config["lr"],
            momentum=optimizer_config["momentum"],
            weight_decay=optimizer_config["weight_decay"],
            dampening=optimizer_config["dampening"],
            nesterov=optimizer_config["nesterov"]
        )
    elif optimizer_config["algorithm"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"]
        )
    else:
        raise ValueError(f"Invalid optimizer configuration: {optimizer_config}")

    scheduler = None
    if scheduler_config is not None:
        if scheduler_config["algorithm"] == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config["T_0"] * len(train_loader)
            )

    metrics = pd.read_csv(metrics_file).values.tolist() if os.path.isfile(metrics_file) else []


    print("Beginning training...")

    for e in range(first_epoch, epochs + 1):

        print(f"Epoch {e}...")

        if scheduler is not None:
            learning_rate = scheduler.get_last_lr()[0]

        train_losses = []
        model.train()

        for data, labels in train_loader:

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
              
            optimizer.zero_grad()

            pred = model(data)

            loss = cross_entropy(pred,labels)
            train_losses.append(loss.item())
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        train_loss = np.mean(train_losses)


        valid_losses = []
        model.eval()

        for data, labels in valid_loader:

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
              
            pred = model(data)
            
            loss = cross_entropy(pred, labels)
            valid_losses.append(loss.item())

        valid_loss = np.mean(valid_losses)


        # print(f"Epoch {e}: {train_loss=}, {valid_loss=}, {learning_rate=}")
        # print(f"Epoch {e}: {train_loss=:0.4f}, {valid_loss=:0.4f}, {learning_rate=:0.4f}")
        # print(f"Epoch {e}: train_loss={train_loss:0.4f}, valid_loss={valid_loss:0.4f}, learning_rate={learning_rate:0.4f}")

        if scheduler is not None:
            print(f"Epoch {e}: train_loss={train_loss:0.4f}, valid_loss={valid_loss:0.4f}, learning_rate={learning_rate:0.4f}")
            metrics.append([e, train_loss, valid_loss, learning_rate])
            pd.DataFrame(
                metrics, columns=["epoch", "train_loss", "valid_loss", "learning_rate"]
            ).to_csv(metrics_file, index=False)
        else:
            print(f"Epoch {e}: train_loss={train_loss:0.4f}, valid_loss={valid_loss:0.4f}")
            metrics.append([e, train_loss, valid_loss])
            pd.DataFrame(
                metrics, columns=["epoch", "train_loss", "valid_loss"]
            ).to_csv(metrics_file, index=False)

        if e in snapshot_epochs:
            print(f"Saving at checkpoint epoch {e}...")
            torch.save(model.state_dict(), f"{snapshots_dir}/snapshot_epoch_{e}.pt")
            # torch.save({
            #     "epoch": e,
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "scheduler_state_dict": scheduler.state_dict(),
            #     "loss": train_loss,
            # }, f"{snapshots_dir}/snapshot_epoch_{e}.pt")

    os.remove("/data/training_inputs.pt")
    os.remove("/data/training_ground_truths.pt")
    os.remove("/data/validation_inputs.pt")
    os.remove("/data/validation_ground_truths.pt")

    print("Finished training.")
    return model


def main():
    
    # Debug model architecture
    model = UnetSegmenter(num_input_channels=2, num_output_channels=3, smallest_layer_depth=64)
    # model.cuda()
    # model.summarize(mode="full")

    from torchsummary import summary
    summary(model.cuda(), (2, 512, 512))


if __name__ == "__main__":
    main()
