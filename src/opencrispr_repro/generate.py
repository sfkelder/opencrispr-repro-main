import json
import os
import re
import csv

import click
import pandas as pd
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from opencrispr_repro.model import ModelSchema, get_model, get_tokenizer

ROOT_DIR = os.path.dirname(__file__)
MAX_LEN = 1500


def kmer_n_repeat(k: int, n: int):
    return "(.)" * k + "".join([f"\\{i+1}" for i in range(k)]) * (n - 1)


def has_kmer_repeats(seq: str):
    k_to_n = [6, 4, 3, 3, 3, 3, 2]  # Thresholds apply to >99.5% of naturals
    return any(re.search(kmer_n_repeat(k, n), seq) for k, n in enumerate(k_to_n, start=1))


def is_valid_seq(seq: str, eos: str):
    return (1000 <= len(seq) <= MAX_LEN) and not has_kmer_repeats(seq) and seq.endswith(eos)


def read_config_file(config_path: str) -> dict:
    with open(config_path) as f:
        if config_path.endswith(".yml") or config_path.endswith(".yaml"):
            config = YAML(typ="safe").load(f)
        else:
            config = json.load(f)
    return config


@click.command()
@click.option("--model-path", "model_path", required=True, help="Path where model is stored")
@click.option("--config", "config_path", required=True, help="JSON or YML w/ generation hyperparams")
@click.option("--save-dir", "save_dir", default='./', help="Path where generated protein sequences are stored")
@click.option("--job-idx", "job_idx", type=str, default=None, required=False, help="Job index (for parallel jobs)")
def main(model_path: str, config_path: str, save_dir: str, job_idx: str):
    config = read_config_file(config_path)
    tokenizer = get_tokenizer(ModelSchema(name="progen2"))

    # File in which to save generations
    gen_dir = os.path.join(save_dir, "generations")
    os.makedirs(gen_dir, exist_ok=True)

    gen_file = os.path.join(gen_dir, os.path.basename(config_path)[:-len(".yml")] + ".csv")
    gen_file = gen_file[:-4] + (f"_{job_idx}.csv" if job_idx is not None else ".csv")

    model = get_model(ModelSchema(name="progen2", path=model_path))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load saved generations if there are any
    if os.path.exists(gen_file):
        all_gens = pd.read_csv(gen_file)
    else:
        all_gens = pd.DataFrame(columns=["context_name", "context", "sequence"])
        with open(gen_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(all_gens.columns)

    # Ensure config is a list
    if isinstance(config, dict):
        config = [config]

    for sub_config in config:
        n_gen, n_kept = 0, 0

        # Configurations
        num_samples = sub_config.pop("num_samples", 10000)
        batch_size = sub_config.pop("batch_size", 10)

        sub_config["do_sample"] = True
        sub_config["num_return_sequences"] = batch_size
        sub_config["temperature"] = sub_config.setdefault("temperature", 1)
        sub_config["top_p"] = sub_config.setdefault("top_p", 1)
        sub_config["top_k"] = sub_config.setdefault("top_k", 0)

        ctx_dict = sub_config.pop("context")
        ctx_name = ctx_dict["name"]
        ctx = ctx_dict["seq"]

        if not ctx.startswith("1") and ctx.endswith("2"):
            ctx = ctx[::-1]
            eos = "1"
        else:
            eos = "2"
        eos_id = tokenizer.encode(eos)[0]

        i = (all_gens.context == ctx).sum().item()

        encoded = tokenizer(
            ctx,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

        # Initialize tqdm progress bar
        pbar = tqdm(total=num_samples, desc=f"Generating {ctx_name}", initial=i)

        while i < num_samples:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    max_length=MAX_LEN,
                    eos_token_id=eos_id,
                    attention_mask=attention_mask,
                    **sub_config,
                )

            gens = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            n_gen += len(gens)
            gens = list(filter(lambda s: is_valid_seq(s, eos), gens))
            n_kept += len(gens)
            gens = gens[: (num_samples - i)]

            with open(gen_file, "a", newline="") as f:
                writer = csv.writer(f)
                for seq in gens:
                    all_gens.loc[len(all_gens)] = {
                        "context_name": f"{ctx_name}_{i}",
                        "context": ctx,
                        "sequence": seq,
                    }
                    writer.writerow([f"{ctx_name}_{i}", ctx, seq])

            i += len(gens)
            pbar.set_description(f"Generations ({n_gen - n_kept}/{n_gen} removed)")
            pbar.update(len(gens))

        pbar.close()


if __name__ == "__main__":
    main()
