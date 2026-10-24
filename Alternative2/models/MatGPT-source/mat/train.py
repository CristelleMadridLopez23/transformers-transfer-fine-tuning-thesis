# Train the decoder:  python train.py --config-name <config>

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig

from utils.dataset import encode, save_vocab_and_model, read_and_preprocess_properties_with_sequences, TextDataset
from utils.models import  TransformerDecoderModel
import os

torch.set_float32_matmul_precision('high')

@hydra.main(config_path="config", config_name="form", version_base=None)
def main(cfg: DictConfig):

    # The runs reported in the paper were not seeded. Set training.seed to make new runs repeatable.
    if cfg.training.get("seed") is not None:
        pl.seed_everything(cfg.training.seed, workers=True)

    tb_logger = TensorBoardLogger(save_dir=cfg.paths.log_dir, name=cfg.training.experiment_name)

    stoi = {
        "H": 0, "He": 1,
        "Li": 2, "Be": 3, "B": 4, "C": 5, "N": 6, "O": 7, "F": 8, "Ne": 9,
        "Na": 10, "Mg": 11, "Al": 12, "Si": 13, "P": 14, "S": 15, "Cl": 16, "Ar": 17,
        "K": 18, "Ca": 19, "Sc": 20, "Ti": 21, "V": 22, "Cr": 23, "Mn": 24, "Fe": 25, "Co": 26, "Ni": 27, "Cu": 28, "Zn": 29,
        "Ga": 30, "Ge": 31, "As": 32, "Se": 33, "Br": 34, "Kr": 35,
        "Rb": 36, "Sr": 37, "Y": 38, "Zr": 39, "Nb": 40, "Mo": 41, "Tc": 42, "Ru": 43, "Rh": 44, "Pd": 45, "Ag": 46, "Cd": 47,
        "In": 48, "Sn": 49, "Sb": 50, "Te": 51, "I": 52, "Xe": 53,
        "Cs": 54, "Ba": 55, "La": 56, "Ce": 57, "Pr": 58, "Nd": 59, "Pm": 60, "Sm": 61, "Eu": 62, "Gd": 63, "Tb": 64, "Dy": 65, "Ho": 66, "Er": 67, "Tm": 68, "Yb": 69, "Lu": 70,
        "Hf": 71, "Ta": 72, "W": 73, "Re": 74, "Os": 75, "Ir": 76, "Pt": 77, "Au": 78, "Hg": 79,
        "Tl": 80, "Pb": 81, "Bi": 82, "Po": 83, "At": 84, "Rn": 85,
        "Fr": 86, "Ra": 87, "Ac": 88, "Th": 89, "Pa": 90, "U": 91, "Np": 92, "Pu": 93, "Am": 94, "Cm": 95, "Bk": 96, "Cf": 97, "Es": 98, "Fm": 99, "Md": 100, "No": 101, "Lr": 102,
        "Rf": 103, "Db": 104, "Sg": 105, "Bh": 106, "Hs": 107, "Mt": 108, "Ds": 109, "Rg": 110, "Cn": 111, "Nh": 112, "Fl": 113, "Mc": 114, "Lv": 115, "Ts": 116, "Og": 117,
        
        # New consolidated edge tokens (27 combinations of +, -, o)
        "ooo": 118, "oo+": 119, "oo-": 120,
        "o+o": 121, "o++": 122, "o+-": 123,
        "o-o": 124, "o-+": 125, "o--": 126,
        "+oo": 127, "+o+": 128, "+o-": 129,
        "++o": 130, "+++": 131, "++-": 132,
        "+-o": 133, "+-+": 134, "+--": 135,
        "-oo": 136, "-o+": 137, "-o-": 138,
        "-+o": 139, "-++": 140, "-+-": 141,
        "--o": 142, "--+": 143, "---": 144,

        # Digits
        "0": 145, "1": 146, "2": 147, "3": 148, "4": 149, 
        "5": 150, "6": 151, "7": 152, "8": 153, "9": 154,

        # Special Tokens
        "<start>": 155, "<end>": 156, "<pad>": 157, 
        "<num>": 158, "<form>": 159, "<band>": 160, "<bulk>": 161
    }

    sequences, properties = read_and_preprocess_properties_with_sequences(
        cfg.data_path.fname,
        cfg.training.selected_properties,
        cfg.training.slices
    )

    # Preserve complete SLICES node indices such as "10" and "11" as tokens.
    index_tokens = sorted(
        {
            token
            for sequence in sequences
            for token in sequence.split()
            if token.isdigit()
        },
        key=int,
    )
    for token in index_tokens:
        if token not in stoi:
            stoi[token] = len(stoi)

    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    special_tokens = [stoi[token] for token in cfg.training.special_tokens[:cfg.training.num_properties]]

    sequence_list = [encode(sequence, stoi, special_tokens) for sequence in sequences]
    max_seq_length = max(len(sequence) for sequence in sequence_list)
    block_size = max_seq_length

    data_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sequence, dtype=torch.long) for sequence in sequence_list],
        batch_first=True,
        padding_value=stoi["<pad>"]
    ) 
    
    n = int(0.9 * len(data_padded))
    train_data, val_data = data_padded[:n], data_padded[n:]

    properties_tensor = torch.tensor(properties, dtype=torch.float)
    train_properties, val_properties = properties_tensor[:n], properties_tensor[n:]

    train_dataset = TextDataset(train_data, train_properties)
    val_dataset = TextDataset(val_data, val_properties) 

    # load the model and finetune
    model = TransformerDecoderModel(
        vocab_size=vocab_size,
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        block_size=block_size,
        dropout=cfg.model.dropout,
        stoi=stoi,
        itos=itos,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=cfg.training.learning_rate,
        batch_size=cfg.training.batch_size,
        num_properties=cfg.training.num_properties,
        warmup_steps=cfg.training.warmup_steps
    )
    
    early_stopping = EarlyStopping('val_loss', patience=2)
    rich_progress_bar = RichProgressBar()
    
    # checkpoint callback - saves best and last model
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename='epoch_{epoch:02d}-val_loss_{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # keep top 3 best checkpoints
        save_last=True,  # always save last checkpoint as 'last.ckpt'
        verbose=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs, 
        accelerator="auto", 
        callbacks=[early_stopping, rich_progress_bar, checkpoint_callback],  
        log_every_n_steps=1, 
        logger=tb_logger
    )
    
    # resume from checkpoint if specified
    ckpt_path = cfg.paths.resume_from_checkpoint if cfg.paths.resume_from_checkpoint else None
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer.fit(model, ckpt_path=ckpt_path)

    # load BEST checkpoint and save its weights for sampling
    best_ckpt = checkpoint_callback.best_model_path
    
    # Ensure weights directory exists BEFORE saving
    os.makedirs(os.path.dirname(cfg.paths.checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.paths.vocab_and_model), exist_ok=True)
    
    if best_ckpt:
        print(f"Loading best checkpoint: {best_ckpt}")
        best_model = TransformerDecoderModel.load_from_checkpoint(
            best_ckpt,
            vocab_size=vocab_size,
            n_embd=cfg.model.n_embd,
            n_head=cfg.model.n_head,
            n_layer=cfg.model.n_layer,
            block_size=block_size,
            dropout=cfg.model.dropout,
            stoi=stoi,
            itos=itos,
            train_dataset=None,
            val_dataset=None,
            learning_rate=cfg.training.learning_rate,
            batch_size=cfg.training.batch_size,
            num_properties=cfg.training.num_properties
        )
        torch.save(best_model.model.state_dict(), cfg.paths.checkpoint_path)
    else:
        # fallback to last model if no best checkpoint found
        torch.save(model.model.state_dict(), cfg.paths.checkpoint_path)
    
    save_vocab_and_model(itos, block_size, cfg.paths.vocab_and_model, stoi, itos)
    print("Training complete.")
    print(f"  - TensorBoard logs: {cfg.paths.log_dir}")
    print(f"  - Checkpoints: {cfg.paths.checkpoint_dir}")
    print(f"  - Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"  - Final weights (from best): {cfg.paths.checkpoint_path}")
    
if __name__ == "__main__":
    main()
