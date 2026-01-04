import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from transformers import Trainer, TrainingArguments, set_seed

from datasets.trajectory_dataset import TrajectoryDataset
from datasets.collate import collate_list_of_sequences

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # 1) 数据
    train_ds = TrajectoryDataset(cfg.data.train_json)
    val_ds   = TrajectoryDataset(cfg.data.val_json)

    # 2) 模型（Hydra 自动递归 instantiate encoder/backbone/decoder）
    model = instantiate(cfg.model)

    # 3) HF Trainer 的参数
    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        num_train_epochs=cfg.train.num_epochs,
        fp16=cfg.train.fp16,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        remove_unused_columns=False,   # 关键：保留我们自定义字段
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_list_of_sequences,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
