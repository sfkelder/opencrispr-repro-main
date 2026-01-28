import logging

import composer.algorithms
from composer import Trainer
from composer.models import HuggingFaceModel
from composer.metrics import LossMetric
from composer.optim import DecoupledAdamW

from .checkpoint import HuggingFaceCheckpointer
from .data import SeqDataset, get_dataloader
from .model import get_model, get_tokenizer
from .scheduler import InvSqrtWithWarmupScheduler
from .schema import FinetuneAPI

logger = logging.getLogger(__name__)


def get_trainer(config: FinetuneAPI):
    tokenizer = get_tokenizer(config.model)
    model = HuggingFaceModel(
        model=get_model(config.model),
        tokenizer=tokenizer,
        eval_metrics=[LossMetric()],
        shift_labels=config.model.name == "progen2",
    )
    logger.info("Initialized model")

    train_data = SeqDataset(
        csv_fname=config.data.train_data_path,
        sequence_col=config.data.sequence_col,
        label_col=config.data.label_col,
    )
    train_dataloader = get_dataloader(config, train_data, tokenizer)
    val_data = SeqDataset(
        csv_fname=config.data.val_data_path,
        sequence_col=config.data.sequence_col,
        label_col=config.data.label_col,
    )
    eval_dataloader = get_dataloader(config, val_data, tokenizer)
    logger.info("Initialized dataloaders")

    train_duration = f"{config.training.train_steps}ba"
    optimizer = DecoupledAdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = InvSqrtWithWarmupScheduler(
        t_warmup=f"{config.training.warmup_steps}ba", t_max=train_duration,
    )
    logger.info("Initialized optimizer")

    algorithms = []
    clipping_threshold = config.training.gradient_clipping_threshold
    if clipping_threshold is not None:
        gradclip = composer.algorithms.GradientClipping(
            clipping_type="norm",
            clipping_threshold=float(clipping_threshold),
        )
        algorithms.append(gradclip)
    
    half = "bfloat16"
    save_interval = f"{config.save_interval_steps}ba"
    checkpointer = HuggingFaceCheckpointer(save_folder=config.save_folder, save_interval=save_interval, precision=half)

    # FSDP setup
    half = "bf16"
    fsdp_config = dict(
        use_orig_params=False,
        limit_all_gathers=True,
        activation_checkpointing=True,
        activation_checkpointing_reentrant=False,
        sync_module_states=True,
        keep_low_precision_grads=False,
        mixed_precision=dict(param_dtype=half, reduce_dtype=half, buffer_dtype=half),
        forward_prefetch=False,
        backward_prefetch="BACKWARD_PRE",
        sharding_strategy="FULL_SHARD",
        state_dict_type="sharded",
        sharded_ckpt_prefix_dir="ba{batch}",
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        spin_dataloaders=False,
        device_train_microbatch_size=config.data.batch_size,
        precision=f"amp_{half}",
        parallelism_config={"fsdp": fsdp_config},
        # Optimization
        max_duration=train_duration,
        optimizers=optimizer,
        schedulers=scheduler,
        algorithms=algorithms,
        step_schedulers_every_batch=True,
        # Save/load
        autoresume=True,
        callbacks=[checkpointer],
        save_folder=config.save_folder,
        save_filename="ba{batch}-rank{rank}.pt",
        save_latest_filename="latest",
        save_overwrite=True,
        save_interval=save_interval,
    )
    logger.info("Initialized trainer")
    return trainer
