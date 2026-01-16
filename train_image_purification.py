import os
import argparse
import sys
import yaml
import torch
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.helpers.dataloader_helper import build_dataloader
from lib.models.monodgp.backbone import build_backbone
from lib.models.monodgp.reconstruction_decoder import ReconstructionDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train reconstruction decoder (feature->RGB)")
    parser.add_argument("--config", required=True, help="YAML config (use dataset/model entries)")
    parser.add_argument("--epochs", type=int, default=100, help="max epochs to run")
    parser.add_argument("--patience", type=int, default=10, help="early-stop patience on no improvement")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="minimum loss drop to count as improvement")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="outputs/reconstruction_decoder.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloader (train split only)
    train_loader, _ = build_dataloader(cfg["dataset"])

    # backbone (frozen)
    backbone = build_backbone(cfg["model"]).to(device)
    backbone.train()

    decoder = ReconstructionDecoder().to(device)
    params = list(backbone.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.L1Loss()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # checkpoint paths
    base_dir = os.path.dirname(args.save_path)
    base_name = os.path.splitext(os.path.basename(args.save_path))[0]
    best_path = os.path.join(base_dir, base_name + "_best.pth")
    last_path = os.path.join(base_dir, base_name + "_last.pth")

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        decoder.train()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True,
            leave=True,
        )
        epoch_loss = 0.0
        for inputs, _, _, _ in pbar:
            inputs = inputs.to(device)

            feats, _ = backbone(inputs)
            # feats is list of NestedTensor sorted by layer name
            if len(feats) < 3:
                raise RuntimeError("Backbone must return layer2/3/4 features.")
            f2 = feats[0].tensors
            f3 = feats[1].tensors
            f4 = feats[2].tensors

            recon = decoder(f2, f3, f4)
            if recon.shape[-2:] != inputs.shape[-2:]:
                recon = torch.nn.functional.interpolate(recon, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(recon, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] avg L1 loss: {avg_loss:.4f}")

        # save best
        if avg_loss + args.min_delta < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "decoder": decoder.state_dict(),
                    "backbone": backbone.state_dict(),
                    "best_loss": best_loss,
                },
                best_path,
            )
            print(f"  -> saved best checkpoint to {best_path}")
        else:
            no_improve += 1

        # early stop if no improvement
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (patience {args.patience})")
            break

    # save last
    torch.save(
        {
            "epoch": epoch + 1,
            "decoder": decoder.state_dict(),
            "backbone": backbone.state_dict(),
            "loss": avg_loss,
        },
        last_path,
    )
    print(f"Saved last checkpoint to {last_path} (best at {best_path})")


if __name__ == "__main__":
    main()
