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

if __name__ == "__main__":
    main()
