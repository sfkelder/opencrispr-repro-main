from pydantic import BaseModel, Field

class DataSchema(BaseModel):
    sequence_col: str
    train_data_path: str
    val_data_path: str
    label_col: str | None = None
    batch_size: int = Field(8)
    num_workers: int = Field(2)

class ModelSchema(BaseModel):
    name: str
    path: str | None = None

    def model_post_init(self, __context):
        assert self.name in ["esm2", "progen2"], self.name
        if self.path is None:
            if self.name == "esm2":
                self.path = "esm2_t33_650M_UR50D"
            elif self.name == "progen2":
                self.path = "base"


class TrainingSchema(BaseModel):
    learning_rate: float = Field(1e-4)
    weight_decay: float | None = Field(None, description="Defaults to learning_rate * 1e-2")
    warmup_steps: int = Field(1000)
    train_steps: int = Field(10000)
    total_lr_decay_factor: float = Field(0.2)
    gradient_clipping_threshold: float = Field(1.0)


class FinetuneAPI(BaseModel):
    save_folder: str
    data: DataSchema
    model: ModelSchema
    training: TrainingSchema
    save_interval_steps: int = Field(1000)
