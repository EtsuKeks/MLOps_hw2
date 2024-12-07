from pathlib import Path
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from dataset import CustomDataModule
from model import CustomModel

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    
    data_module = CustomDataModule(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    
    model = CustomModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_size,
        output_size=cfg.model.output_size,
        learning_rate=cfg.model.learning_rate
    )
    
    trainer = Trainer(**cfg.trainer)
    trainer.fit(model, data_module)

    output_dir = Path(hydra.utils.to_absolute_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.ckpt"
    trainer.save_checkpoint(model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
