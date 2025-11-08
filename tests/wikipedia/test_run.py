from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from dit import DiT_Llama
from mdlm import MDLM
import os
from datetime import datetime
import math

# Hyperparameters for training
BATCH_SIZE = 64
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 1e-5
MAX_SEQ_LEN_FOR_BATCH = 1024 # Start with 1024
WANDB_LOG = True

# Hyperparameters for validation
N_VALIDATION_SAMPLES = 10_000 # Validation set size

# Hyperparemeters for sampling
NUM_SAMPLES = 1
SAMPLING_STEPS = 200
SAVE_SAMPLE_AS_FILE = True

def run_wikipedia_training():    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get tokenizer
    MODEL_NAME = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # BERT tokenizer (Need [MASK] token!)
    
    # Load dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    chunk_data = dataset["train"].shuffle(buffer_size=10_000, seed=42)
    data_set_name = chunk_data.info.dataset_name
    
    # Tokenize dataset
    tokenized_dataset = chunk_data.map(
        lambda example:  tokenizer(
            example['text'],
            truncation=True,
            max_length=MAX_SEQ_LEN_FOR_BATCH,
            return_overflowing_tokens=True,
            stride=256,
            padding='max_length',
        ),
        batched=True,
        remove_columns=['text', 'id', 'url', 'title'],
    )

    # Split data into train and validation sets
    validation_step_cnt = math.ceil(N_VALIDATION_SAMPLES // BATCH_SIZE)
    validation_data = tokenized_dataset.take(N_VALIDATION_SAMPLES)
    train_data = tokenized_dataset.skip(N_VALIDATION_SAMPLES)

    # Generate train data iterator for the training.
    train_iterator = train_data.iter(batch_size=BATCH_SIZE)
    validation_iterator = validation_data.iter(batch_size=BATCH_SIZE)

    # Instantiate models, optimizer, and loss
    model = DiT_Llama(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN_FOR_BATCH,
        dim=256,
        n_layers=10,
        n_heads=8,
    ).to(device)
    mdlm = MDLM(model, tokenizer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loggin on wandb
    if WANDB_LOG:
        wandb.init(
            project=f"min_mdlm_{data_set_name}",
            config={
                "learning_rate": LEARNING_RATE,
                "architecture": "DiT_Llama",
                "dataset": data_set_name,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
            }
        )            
    
    # Setup for saving samples
    if SAVE_SAMPLE_AS_FILE:
        samples_dir = "samples"
        os.makedirs(samples_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"{data_set_name}_samples_{timestamp}.txt"
        file_to_write = os.path.join(samples_dir, file_name)
        
        with open(file_to_write, "w", encoding="utf-8") as f:
            f.write(f"Settings: Dataset={data_set_name}, Epochs={NUM_EPOCHS}, Steps per Epoch={STEPS_PER_EPOCH}, Sampling Steps={SAMPLING_STEPS}\n\n\n")
        

    print(f"Start training: Total {NUM_EPOCHS} epochs, {STEPS_PER_EPOCH} steps per epoch")
    for epoch in range(NUM_EPOCHS):
        # Train!
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} training begins.")
        mdlm.train()
        total_train_loss = 0.0
        t_max_epoch = 0.0
        t_max_step = 0.0
        pbar_train = tqdm(range(STEPS_PER_EPOCH))
        for step in pbar_train:
            try:
                batch_tokens = next(train_iterator)

            except StopIteration:
                print("Test dataset stream is over. Reset the iterator.")
                train_iterator = train_data.iter(batch_size=BATCH_SIZE)
                break

            x = torch.tensor(batch_tokens['input_ids'], device=device)
            attention_mask = torch.tensor(batch_tokens['attention_mask'], device=device)
            
            if x.ndim == 3 and BATCH_SIZE == 1:
                x = x.squeeze(0)
                attention_mask = attention_mask.squeeze(0)

            # Draw random time step t
            t = torch.rand((BATCH_SIZE,), device=x.device)
            t_max_step = t.max().item() if t.max().item() > t_max_step else t_max_step
            
            ce_loss = mdlm(x, t, attention_mask)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            # pbar.set_description(f"CE Loss: {ce_loss.item():.4f}")

            if WANDB_LOG and step % 10 == 0:
                wandb.log({"train_step_ce_loss": ce_loss.item()})
                wandb.log({"max t": t_max_step})
            total_train_loss += ce_loss.item()
            pbar_train.set_description(f"Avg Train Loss: {total_train_loss / (step+1):.4f}")

        t_max_epoch = t_max_step if t_max_step > t_max_epoch else t_max_epoch
        avg_epoch_train_loss = total_train_loss / (step+1)
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} training ends : {avg_epoch_train_loss:.4f}, max t : {t_max_epoch:.4f}")
        if WANDB_LOG:
            wandb.log({"avg_epoch_train_loss": avg_epoch_train_loss, "epoch": epoch+1})           
            wandb.log({"epoch_max_t": t_max_epoch, "epoch": epoch+1})          
            

        # Validation
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} validation begins.")
        mdlm.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            # Reset validation iterator
            validation_iterator = validation_data.iter(batch_size=BATCH_SIZE)
            pbar_val = tqdm(range(validation_step_cnt))
            for val_step in pbar_val:
                try:
                    val_batch_tokens = next(validation_iterator)

                except StopIteration:
                    print("Valiation dataset stream is over. Reset the iterator.")
                    validation_iterator = validation_data.iter(batch_size=BATCH_SIZE)
                    break

                val_x = torch.tensor(val_batch_tokens['input_ids'], device=device)
                val_attention_mask = torch.tensor(val_batch_tokens['attention_mask'], device=device)
                
                if val_x.ndim == 3 and BATCH_SIZE == 1:
                    val_x = val_x.squeeze(0)
                    val_attention_mask = val_attention_mask.squeeze(0)

                val_t = torch.rand((BATCH_SIZE,), device=val_x.device)
                
                val_ce_loss = mdlm(val_x, val_t, val_attention_mask)
                total_val_loss += val_ce_loss.item()
                pbar_val.set_description(f"Avg Val Loss: {total_val_loss / (val_step+1):.4f}")

        avg_epoch_val_loss = total_val_loss / (val_step+1)
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} validation ends : {avg_epoch_val_loss:.4f}")
        if WANDB_LOG:
            wandb.log({"avg_epoch_val_loss": avg_epoch_val_loss, "epoch": epoch+1})
        
        
        with torch.no_grad():
            # Sampling
            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} sampling begins.")
            sampled_token_seq = mdlm.sample(NUM_SAMPLES, SAMPLING_STEPS, device=device)
            sampled_texts = tokenizer.batch_decode(sampled_token_seq, skip_special_tokens=False)
        
            if SAVE_SAMPLE_AS_FILE:                
                try:
                    with open(file_to_write, "a", encoding="utf-8") as f:
                        f.write(f"--- Epoch {epoch+1} Samples ---\n")
                        for i, text in enumerate(sampled_texts):
                            f.write(f"[Sample {i+1}]\n{text}\n")  # Create or clear the file
                        f.write("\n")
                        f.flush()
                except Exception as e:
                    print(f"Error saving samples: {e}")
                    
            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} sampling finished.")


    print(f"End of the training: Total {NUM_EPOCHS} epochs.")
    if WANDB_LOG:
        wandb.finish()