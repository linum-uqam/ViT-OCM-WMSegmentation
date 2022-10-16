# Imports

from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Resize
from datasets import load_metric
from transformers import SwinForImageClassification
from transformers import SwinModel, SwinConfig
from transformers import AutoFeatureExtractor
from transformers import default_data_collator
from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
import numpy as np
import torch
import os
import argparse

# Global Variables
# pre-trained model from which to fine-tune
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
BATCH_SIZE = 16  # batch size for training and evaluation


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description="Train a model on Allen data")
    # data path
    parser.add_argument("--data_path", type=str,
                        default="../Allen_data/Splited_data.zip", help="Path to the data")
    # model name
    parser.add_argument("--model_name", type=str,
                        default="microsoft/swin-tiny-patch4-window7-224", help="Name of the model to use")
    # batch size
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Data loading
    dataset = load_dataset("imagefolder", data_files=args.data_path)

    # split up training into training + validation
    splits = dataset["train"].train_test_split(test_size=0.2)
    train_ds = splits['train']
    val_ds = splits['test']

    # Model and utils loading
    metric = load_metric("accuracy")

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    config = SwinConfig(num_labels=5,
                        label2id=label2id,
                        id2label=id2label)

    model_without_Pretrained = SwinModel(config)

    model_without_Pretrained = SwinForImageClassification(
        config)

    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        label2id=label2id,
        id2label=id2label,
        # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ignore_mismatched_sizes=True,
    )

    train_transforms = Compose(
        [
            # RandomResizedCrop(feature_extractor.size),
            Resize(feature_extractor.size),
            RandomHorizontalFlip(),
            ToTensor(),
            # normalize,
        ]
    )

    val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            # CenterCrop(feature_extractor.size),
            ToTensor(),
            # normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(
            image.convert("RGB")) for image in example_batch["image"]]
        return example_batch

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)

    # the compute_metrics function takes a Named Tuple as input:
    # predictions, which are the logits of the model as Numpy arrays,
    # and label_ids, which are

    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"]
                                   for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    args = TrainingArguments(
        f"{args.model_name}-allen",
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    train_results = trainer.train()
    # rest is optional but nice to have
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    # some nice to haves:
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


# pip install torch & pip install torchvision &pip install numpy & pip install transformers & pip install timm & pip install datasets & pip install -U albumentations opencv-python
