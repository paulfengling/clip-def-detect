import os
import argparse
import json
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from utils import setup_seed
from model.adapter import AdaptedCLIP
from model.clip import create_model
from model.runtime import resolve_device, empty_device_cache
from dataset import get_dataset
from forward_utils import (
    get_adapted_text_embedding,
    get_adapted_single_class_text_embedding,
    calculate_similarity_map,
    calculate_seg_loss,
)
import warnings

warnings.filterwarnings("ignore")

cpu_num = 4

os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_text_adapter(
    adapted_model: nn.Module,
    clip_surgery: nn.Module,
    text_norm_weight: float,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    # scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    text_epoch: int,
    dataset_name: str,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, text_epoch):
        logger.info(f"training text epoch {epoch}:")

        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            class_names = input_data["class_name"]

            # forward text
            epoch_text_feature_dict = {}
            for class_name in list(set(class_names)):
                text_embedding = get_adapted_single_class_text_embedding(
                    adapted_model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding
            epoch_text_feature = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # bs,768,2
            # forward image
            with torch.no_grad():
                _, patch_features = clip_surgery.encode_image(image, adapted_model.levels)
                cls_token, _ = adapted_model.clipmodel.encode_image(image, [])
                cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
                patch_features = [
                    clip_surgery.visual.ln_post(t[:, 1:, :]) for t in patch_features
                ]
                patch_features = [t @ clip_surgery.visual.proj for t in patch_features]
                patch_features = [
                    t / t.norm(dim=-1, keepdim=True) for t in patch_features
                ]
                patch_features = [t + cls_token.unsqueeze(1) for t in patch_features]
            # calculate similarity and get prediction
            loss = 0.0
            for f in patch_features:
                # bs,patch_num,768
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)
            orthogonal_loss = (
                (epoch_text_feature[:, :, 0] * epoch_text_feature[:, :, 1])
                .sum(1)
                .mean()
            ) ** 2
            loss += orthogonal_loss * text_norm_weight
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        ckp_path = os.path.join(save_path, "text_adapter.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "text_adapter": adapted_model.text_adapter.state_dict(),
                "text_optimizer": optimizer.state_dict(),
            },
            ckp_path,
        )
    return adapted_model


def train_image_adapter(
    model: nn.Module,
    text_embeddings: torch.Tensor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    start_epoch: int,
    save_path: str,
    image_epoch: int,
    img_size: int,
    logger: logging.Logger,
):
    for epoch in range(start_epoch, image_epoch):
        logger.info(f"training image epoch {epoch}:")
        loss_list = []
        for input_data in tqdm(train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            B, C, H, W = image.shape
            # forward text
            class_names = input_data["class_name"]
            epoch_text_feature = torch.stack(
                [text_embeddings[class_name] for class_name in class_names], dim=0
            )

            # forward image
            patch_features, det_feature = model(image)
            # calculate similarity and get prediction
            loss = 0.0
            det_feature = det_feature.unsqueeze(1)
            cls_preds = torch.matmul(det_feature, epoch_text_feature)[:, 0]
            loss += F.cross_entropy(cls_preds, label)
            for f in patch_features:
                # text-image alignment
                patch_preds = calculate_similarity_map(f, epoch_text_feature, img_size)
                loss += calculate_seg_loss(patch_preds, mask)  # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            scheduler.step()
        logger.info(f"loss: {np.mean(loss_list)}")
        # save checkpoint
        model_dict = {
            "epoch": epoch + 1,
            "image_adapter": model.image_adapter.state_dict(),
            "image_optimizer": optimizer.state_dict(),
            "image_scheduler": scheduler.state_dict(),
        }
        torch.save(model_dict, os.path.join(save_path, "image_adapter.pth"))
        if (epoch + 1) % 1 == 0:
            ckp_path = os.path.join(save_path, f"image_adapter_{epoch + 1}.pth")
            torch.save(
                model_dict,
                ckp_path,
            )
    return model


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--quick_gelu", action=argparse.BooleanOptionalAction, default=None,
                        help="Use OpenAI QuickGELU (default for new training; resumed runs preserve their setting)")
    parser.add_argument("--surgery_until_layer", type=int, default=20)
    parser.add_argument("--relu", action="store_true", help="use relu after projection")
    # training
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--meta_path", type=str, help="Explicit training JSONL, overrides --shot selection")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_checkpointing", action="store_true", help="Reduce activation memory")
    parser.add_argument(
        "--training_mode",
        type=str,
        default="few_shot",
        choices=["few_shot", "full_shot"],
    )
    parser.add_argument("--shot", type=int, default=32, help="number of shots (0 means full shot)")
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--text_epoch", type=int, default=1, help="epochs for stage1")
    parser.add_argument("--image_epoch", type=int, default=4, help="epochs for stage2")
    parser.add_argument("--text_lr", type=float, default=0.00001, help="learning rate for stage1")
    parser.add_argument("--image_lr", type=float, default=0.0005, help="learning rate for stage2")
    parser.add_argument(
        "--criterion", type=str, default=["dice_loss", "focal_loss"], nargs="+"
    )
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    # hyper-parameters
    parser.add_argument("--text_norm_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)

    args = parser.parse_args()
    if args.img_size < 56 or args.img_size % 14:
        parser.error("--img_size must be a multiple of 14, at least 56")
    if args.text_epoch < 0 or args.image_epoch < 1:
        parser.error("--text_epoch must be >= 0 and --image_epoch must be >= 1")
    if args.text_batch_size < 1 or args.image_batch_size < 1 or args.num_workers < 0:
        parser.error("Batch sizes must be positive and --num_workers must be >= 0")
    if args.shot < 0:
        parser.error("--shot must be >= 0 (0 selects full-shot training)")
    if not 0 <= args.text_adapt_until <= 12 or not 0 <= args.image_adapt_until <= 24:
        parser.error("Text/image adaptation layers must be within 0..12 / 0..24")
    if not 1 <= args.surgery_until_layer <= 25:
        parser.error("--surgery_until_layer must be within 1..25")
    if args.shot == 0:
        args.training_mode = "full_shot"
    try:
        device = resolve_device(args.device)
    except ValueError as error:
        parser.error(str(error))
    # ========================================================
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.addHandler(logging.StreamHandler())
    setup_seed(args.seed, device=device, logger=logger)
    config_path = os.path.join(args.save_path, "config.json")
    previous = None
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            previous = json.load(handle)
    if args.quick_gelu is None:
        args.quick_gelu = previous.get("quick_gelu", False) if previous is not None else True
    if previous is not None:
        for key in ("model_name", "img_size", "relu", "text_adapt_weight", "image_adapt_weight",
                    "text_adapt_until", "image_adapt_until", "text_epoch", "dataset", "data_root", "meta_path",
                    "surgery_until_layer", "text_norm_weight", "seed", "training_mode", "shot", "quick_gelu"):
            old_value = previous.get(key, False) if key == "quick_gelu" else previous.get(key)
            if old_value != vars(args).get(key):
                raise ValueError(f"Resume setting {key} changed; use a new --save_path")
    logger.info("args: %s", vars(args))
    logger.info("Training device: %s", device)
    text_dataset, image_dataset = get_dataset(
        args.dataset, args.img_size, args.training_mode, args.shot, "train", logger,
        data_path=args.data_root, meta_path=args.meta_path,
    )
    args.class_names = sorted({row["class_name"] for row in image_dataset.meta})
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    use_cuda = device.type == "cuda"
    # ========================================================
    # load model
    # set up model for training
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        force_quick_gelu=args.quick_gelu,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)
    model.eval()
    model.grad_checkpointing = args.grad_checkpointing
    # set optimizer
    text_optimizer = torch.optim.Adam(
        model.text_adapter.parameters(),
        lr=args.text_lr,
        betas=(0.5, 0.999),
    )
    image_optimizer = torch.optim.Adam(
        model.image_adapter.parameters(),
        lr=args.image_lr,
        betas=(0.5, 0.999),
    )
    # text_scheduler = MultiStepLR(text_optimizer, milestones=[400], gamma=0.1)
    image_scheduler = MultiStepLR(image_optimizer, milestones=[16000, 32000], gamma=0.5)
    # ========================================================
    # load checkpoints if exists
    text_file = glob(args.save_path + "/text_adapter.pth")
    if len(text_file) > 0:
        checkpoint = torch.load(text_file[0], map_location=device, weights_only=True)
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        text_optimizer.load_state_dict(checkpoint["text_optimizer"])
        text_start_epoch = checkpoint["epoch"]
        adapt_text = args.text_epoch > 0 and text_start_epoch < args.text_epoch
    elif args.text_epoch == 0:
        adapt_text = False
    else:
        text_start_epoch = 0
        adapt_text = True  # check if text adapter is loaded
    file = glob(args.save_path + "/image_adapter.pth")
    if len(file) > 0:
        checkpoint = torch.load(file[0], map_location=device, weights_only=True)
        image_start_epoch = checkpoint["epoch"]
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        image_optimizer.load_state_dict(checkpoint["image_optimizer"])
        if "image_scheduler" in checkpoint:
            image_scheduler.load_state_dict(checkpoint["image_scheduler"])
    else:
        image_start_epoch = 0
    if not adapt_text and image_start_epoch >= args.image_epoch:
        logger.info("Requested epochs already completed (image epoch %d); increase --image_epoch to continue", image_start_epoch)
        return
    # ========================================================
    # load dataset
    if args.training_mode == "full_shot":
        args.shot = -1
    kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}
    logger.info("loading image adaptation dataset ...")
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=args.image_batch_size, shuffle=True, **kwargs
    )
    # ========================================================
    # training
    if adapt_text:
        # The surgery model is needed only during unfinished text adaptation.
        # Avoid allocating a second large CLIP when skipping/resuming stage 2.
        clip_surgery = create_model(
            model_name=args.model_name, img_size=args.img_size, device=device,
            pretrained="openai", require_pretrained=True,
            force_quick_gelu=args.quick_gelu,
        )
        clip_surgery.visual.DAPM_replace(DPAM_layer=args.surgery_until_layer)
        clip_surgery.eval().requires_grad_(False)
        text_dataloader = torch.utils.data.DataLoader(
            text_dataset, batch_size=args.text_batch_size, shuffle=True, **kwargs
        )
        model = train_text_adapter(
            adapted_model=model,
            clip_surgery=clip_surgery,
            text_norm_weight=args.text_norm_weight,
            train_loader=text_dataloader,
            optimizer=text_optimizer,
            # scheduler=text_scheduler,
            device=device,
            start_epoch=text_start_epoch,
            dataset_name=args.dataset,
            save_path=args.save_path,
            text_epoch=args.text_epoch,
            img_size=args.img_size,
            logger=logger,
        )
        del text_dataloader, clip_surgery
    del text_dataset, text_optimizer
    empty_device_cache(device)
    with torch.no_grad():
        if args.text_epoch == 0:
            text_embeddings = get_adapted_text_embedding(
                clip_model, args.dataset, device
            )
        else:
            text_embeddings = get_adapted_text_embedding(model, args.dataset, device)
    model = train_image_adapter(
        model=model,
        text_embeddings=text_embeddings,
        image_epoch=args.image_epoch,
        train_loader=image_dataloader,
        optimizer=image_optimizer,
        scheduler=image_scheduler,
        device=device,
        start_epoch=image_start_epoch,
        save_path=args.save_path,
        img_size=args.img_size,
        logger=logger,
    )


if __name__ == "__main__":
    main()
