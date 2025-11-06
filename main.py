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

# Hyperparameters for training
BATCH_SIZE = 64
NUM_EPOCHS = 10
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 5e-4
MAX_SEQ_LEN_FOR_BATCH = 1024 # Start with 1024
WANDB_LOG = True

# Hyperparemeters for sampling
NUM_SAMPLES = 5
SAMPLING_STEPS = 100
SAVE_SAMPLE_AS_FILE = True

if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get tokenizer
    MODEL_NAME = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   # BERT tokenizer (Need [MASK] token!)
    
    # Load dataset
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
    chunk_data = dataset["train"].shuffle(buffer_size=10_000, seed=42)
    data_set_name = chunk_data.info.dataset_name

    # Split data
    N_VALIDATION_SAMPLES = 10_000 # Validation set size
    validation_data = chunk_data.take(N_VALIDATION_SAMPLES)
    train_data = chunk_data.skip(N_VALIDATION_SAMPLES)
    print(f"Train Dataset : {train_data}")
    print(f"Validation Dataset : {validation_data}")

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
        pbar_train = tqdm(range(STEPS_PER_EPOCH))
        for step in pbar_train:
            try:
                batch_raw = next(train_iterator)

            except StopIteration:
                print("Dataset stream is over. Reset the iterator.")
                train_iterator = train_data.iter(batch_size=BATCH_SIZE)
                break

            text_list = [text for text in batch_raw['text'] if text is not None]
            if not text_list:
                print("Skipping empty batch (all samples were None)")
                continue

            # Tokenize the data
            tokenized_batch = tokenizer(
                text_list, # Flatten the list of lists : First str items only!
                padding='max_length', # Pad emptyslots! (Distinguish MASK and Empty)
                truncation=True,      # Trunc if longer than max_length
                max_length=MAX_SEQ_LEN_FOR_BATCH,
                return_tensors='pt',  # pytorch tensor!
            )

            x = tokenized_batch['input_ids'].to(device)
            attention_mask = tokenized_batch['attention_mask'].to(device)

            ce_loss = mdlm(x, attention_mask)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            # pbar.set_description(f"CE Loss: {ce_loss.item():.4f}")

            if WANDB_LOG:
                wandb.log({"train_step_ce_loss": ce_loss.item()})
            total_train_loss += ce_loss.item()
            pbar_train.set_description(f"Avg Train Loss: {total_train_loss / (step+1):.4f}")

        avg_epoch_train_loss = total_train_loss / (step+1)
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} training ends : {avg_epoch_train_loss:.4f}")
        if WANDB_LOG:
            wandb.log({"avg_epoch_train_loss": avg_epoch_train_loss, "epoch": epoch+1})            
            

        # Validation
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} validation begins.")
        mdlm.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            pbar_val = tqdm(range(STEPS_PER_EPOCH))
            for val_step in pbar_val:
                try:
                    val_batch_raw = next(validation_iterator)

                except StopIteration:
                    print("Dataset stream is over. Reset the iterator.")
                    validation_iterator = validation_data.iter(batch_size=BATCH_SIZE)
                    break
            
                val_text_list = [text for text in val_batch_raw['text'] if text is not None]
                if not val_text_list:
                    print("Skipping empty batch (all samples were None)")
                    continue
                
                val_tokenized_batch = tokenizer(
                    val_text_list,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_SEQ_LEN_FOR_BATCH,
                    return_tensors='pt',
                )

                val_x = val_tokenized_batch['input_ids'].to(device)
                val_attention_mask = val_tokenized_batch['attention_mask'].to(device)

                val_ce_loss = mdlm(val_x, val_attention_mask)
                total_val_loss += val_ce_loss.item()
                pbar_val.set_description(f"Avg Val Loss: {total_val_loss / (val_step+1):.4f}")

            avg_epoch_val_loss = total_val_loss / (val_step+1)
            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} validation ends : {avg_epoch_val_loss:.4f}")
            if WANDB_LOG:
                wandb.log({"avg_epoch_val_loss": avg_epoch_val_loss, "epoch": epoch+1})
        
        
            # Sampling
            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} sampling begins.")
            sampled_token_seq = mdlm.sample(NUM_SAMPLES, SAMPLING_STEPS, device=device)
            sampled_texts = tokenizer.batch_decode(sampled_token_seq, skip_special_tokens=False)
        
            if SAVE_SAMPLE_AS_FILE:                
                try:
                    with open(file_to_write, "a", encoding="utf-8") as f:
                        f.write(f"--- Epoch {epoch+1} Samples ---\n")
                        for i, text in enumerate(sampled_texts):
                            f.write(f"Sample {i+1} : {text}\n")  # Create or clear the file
                        f.write("\n")
                        f.flush()
                except Exception as e:
                    print(f"Error saving samples: {e}")
                    
            print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} sampling finished.")


    print(f"End of the training: Total {NUM_EPOCHS} epochs.")
    if WANDB_LOG:
        wandb.finish()