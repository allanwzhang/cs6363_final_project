# testing/clip_zero_shot_eval.py

import torch
import torch.nn.functional as F
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform


CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

PROMPTS = {
    "Smiling": [
        "a photo of a smiling person",
        "a photo of a person who is not smiling",
    ],
    "Eyeglasses": [
        "a photo of a person wearing eyeglasses",
        "a photo of a person not wearing eyeglasses",
    ],
    "Blond_Hair": [
        "a photo of a person with blond hair",
        "a photo of a person without blond hair",
    ],
    "Male": [
        "a photo of a male person",
        "a photo of a female person",
    ],
    "Wearing_Hat": [
        "a photo of a person wearing a hat",
        "a photo of a person not wearing a hat",
    ],
}


def get_attr_index(attr_name: str) -> int:
    return CELEBA_ATTR_NAMES.index(attr_name)


def celeba_to_binary(attr_tensor: torch.Tensor) -> torch.Tensor:
    # CelebA usually stores attributes as -1 / +1
    return (attr_tensor > 0).long()


def compute_binary_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


@torch.no_grad()
def evaluate_attribute(
    text_model,
    vision_model,
    tokenizer,
    loader,
    attr_name: str,
    device: str = "cpu",
    max_batches: int | None = None,
):
    if attr_name not in PROMPTS:
        raise ValueError(f"No prompts defined for attribute: {attr_name}")

    attr_idx = get_attr_index(attr_name)
    prompt_texts = PROMPTS[attr_name]

    text_inputs = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    text_outputs = text_model(**text_inputs)
    text_features = F.normalize(text_outputs.text_embeds, dim=-1)  # [2, d]

    all_preds = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)

        # Change this if your dataloader uses a different key
        # e.g. batch["attr"] instead of batch["attributes"]
        attrs = batch["target"]
        labels = celeba_to_binary(attrs[:, attr_idx]).to(device)

        image_outputs = vision_model(pixel_values=images)
        image_features = F.normalize(image_outputs.image_embeds, dim=-1)  # [B, d]

        logits = image_features @ text_features.T  # [B, 2]
        pred_is_positive = (logits[:, 0] > logits[:, 1]).long()

        all_preds.append(pred_is_positive.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return compute_binary_metrics(all_labels, all_preds)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = get_basic_image_transform(224)

    _, loader = build_celeba_dataloader(
        root="./datasets",
        split="test",
        transform=transform,
        batch_size=32,
    )

    model_name = "openai/clip-vit-base-patch32"

    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)

    text_model.eval()
    vision_model.eval()

    attributes_to_test = ["Smiling", "Eyeglasses", "Blond_Hair"]

    for attr_name in attributes_to_test:
        metrics = evaluate_attribute(
            text_model=text_model,
            vision_model=vision_model,
            tokenizer=tokenizer,
            loader=loader,
            attr_name=attr_name,
            device=device,
            max_batches=100,  # set to None for full test set
        )

        print(f"\n=== {attr_name} ===")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1       : {metrics['f1']:.4f}")
        print(f"Confusion Matrix [ [TN, FP], [FN, TP] ]: {metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()