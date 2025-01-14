from typing import TypedDict, Literal

class ModelConfig(TypedDict):
	d_model: int
	nhead: int
	num_layers: int
	seq_length: int
	dropout: float
	device: str

class DataConfig(TypedDict):
	data_dir: str
	train_split: float
	val_split: float
	type: Literal['mixed', 'pov', 'full']
	max_logs: int | None

class TrainingConfig(TypedDict):
	batch_size: int
	epochs: int
	learning_rate: float
	weight_decay: float
	save_every: int
	checkpoint_path: str | None
	num_workers: int
	prefetch_factor: int

class GenerationConfig(TypedDict):
	temperature: float
	max_tokens: int
	checkpoint_path: str
	prompt: str

class Config(TypedDict):
	model: ModelConfig
	data: DataConfig
	training: TrainingConfig
	generation: GenerationConfig