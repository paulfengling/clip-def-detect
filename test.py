import os
import argparse
import numpy as np
from tqdm import tqdm
import logging
from glob import glob
from pandas import DataFrame, Series
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from utils import setup_seed, cos_sim
from custom_infer import Predictor
from model.adapter import AdaptedCLIP
from model.clip import create_model
from dataset import get_dataset, DOMAINS
from forward_utils import (
    get_adapted_text_embedding,
    calculate_similarity_map,
    metrics_eval,
    visualize,
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


def get_support_features(model, support_loader, device):
    all_features = []
    for input_data in support_loader:  # bs always=1. training for an epoch first, Then use this updated model for memory bank construction.
        image = input_data[0].to(device)
        patch_tokens = model(image)
        patch_tokens = [t.reshape(-1, 768) for t in patch_tokens]
        all_features.append(patch_tokens)
    support_features = [
        torch.cat([all_features[j][i] for j in range(len(all_features))], dim=0)
        for i in range(len(all_features[0]))
    ]
    return support_features


def get_predictions(
    model: nn.Module,
    class_text_embeddings: torch.Tensor,
    test_loader: DataLoader,
    device: str,
    img_size: int,
    dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].cpu().numpy()
        label = input_data["label"].cpu().numpy()
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_feature = class_text_embeddings
        # forward image
        patch_features, det_feature = model(image)
        # calculate similarity and get prediction
        # cls_preds = []
        pred = det_feature @ epoch_text_feature
        pred = (pred[:, 1] + 1) / 2
        preds_image.append(pred.cpu().numpy())
        patch_preds = []
        for f in patch_features:
            # f: bs,patch_num,768
            patch_pred = calculate_similarity_map(
                f, epoch_text_feature, img_size, test=True, domain=DOMAINS[dataset]
            )
            patch_preds.append(patch_pred)
        patch_preds = torch.cat(patch_preds, dim=1).sum(1).cpu().numpy()
        preds.append(patch_preds)
    masks = np.concatenate(masks, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.concatenate(preds, axis=0)
    preds_image = np.concatenate(preds_image, axis=0)
    return masks, labels, preds, preds_image, file_names


def main():
    parser = argparse.ArgumentParser(description="Training")
    # model
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    # testing
    parser.add_argument("--dataset", type=str, default="MVTec")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--meta_path", type=str, help="Held-out evaluation JSONL")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shot", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    # exp
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--save_path", type=str, default="ckpt/baseline")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch_size must be positive and --num_workers must be >= 0")
    # ========================================================
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
    )
    logger.info("args: %s", vars(args))
    # ========================================================
    # load model
    # set up model for testing
    predictor = Predictor(args.save_path, args.img_size, args.device)
    device = predictor.device
    setup_seed(args.seed, device=device, logger=logger)
    use_cuda = device.type == "cuda"
    logger.info("Evaluation device: %s", device)
    model = predictor.model
    clip_model = model.clipmodel
    adapt_text = predictor.adapt_text
    args.img_size = predictor.img_size
    files = [predictor.checkpoint]
    for file in files:
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        test_epoch = checkpoint["epoch"]
        print("load image adapter -->> ", file)
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        # ========================================================
        # load dataset
        kwargs = {"num_workers": args.num_workers, "pin_memory": use_cuda}
        image_datasets = get_dataset(
            args.dataset,
            args.img_size,
            None,
            args.shot,
            "test",
            logger=logger,
            data_path=args.data_root,
            meta_path=args.meta_path,
        )
        with torch.no_grad():
            if adapt_text:
                text_embeddings = get_adapted_text_embedding(
                    model, args.dataset, device
                )
            else:
                text_embeddings = get_adapted_text_embedding(
                    clip_model, args.dataset, device
                )
        # ========================================================
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        score_rows = []
        for class_name, image_dataset in image_datasets.items():
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
            )

            # ========================================================
            # testing
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    img_size=args.img_size,
                    dataset=args.dataset,
                )
            # ========================================================
            threshold = predictor.thresholds.get(class_name)
            for filename, label, score, pixel_map in zip(file_names, labels, preds_image, preds):
                row = {"image": filename, "class_name": class_name, "label": int(label),
                       "image_score": float(score), "pixel_max": float(pixel_map.max())}
                if threshold is not None:
                    row.update(threshold=threshold, score_type=predictor.score_type,
                               predicted_label=int(row[predictor.score_type] >= threshold))
                score_rows.append(row)
            if args.visualize:
                visualize(
                    masks,
                    preds,
                    file_names,
                    args.save_path,
                    args.dataset,
                    class_name=class_name,
                )
            class_result_dict = metrics_eval(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)
        average = df.drop(columns="class name").mean().to_dict()
        average["class name"] = "Average"
        df.loc[len(df)] = Series(average)
        df.to_csv(os.path.join(args.save_path, "metrics.csv"), index=False)
        DataFrame(score_rows).to_csv(os.path.join(args.save_path, "image_scores.csv"), index=False)
        print(df.to_string(index=False))
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
        break

if __name__ == "__main__":
    main()
