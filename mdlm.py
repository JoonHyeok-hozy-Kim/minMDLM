import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

class MDLM(nn.Module):
  def __init__(
      self, 
      model,
      tokenizer,
  ):
    super().__init__()
    self.model = model
    self.tokenizer = tokenizer
    self.loss_function = nn.CrossEntropyLoss(
      ignore_index=self.tokenizer.pad_token_id,
      reduction='none'  # To weight with alpha_t_prime / (1-alpha_t)
    )

  def forward(self, x, attention_mask):
    batch_size, seq_len = x.shape

    # Sample time
    t = torch.rand((batch_size,), device=x.device)

    # Draw latent z_t from the Categorical distribution and update attention_mask
    z_t, diffusion_mask, loss_weight = self.add_noise(x, t, attention_mask, "cosine")
    loss_weight = loss_weight.detach() # Negative value

    # Get x_theta
    x_theta = self.model(z_t, t, attention_mask)
    # print(f"x.shape : {x.shape}, z_t.shape : {z_t.shape}, x_theta.shape : {x_theta.shape}")

    # Get Cross Entropy Loss
    logits_flat = x_theta.view(-1, self.tokenizer.vocab_size)  # (batch_size*max_seq_len, vocab_size)
    targets = x.clone()
    targets[~diffusion_mask] = self.tokenizer.pad_token_id
    targets_flat = targets.view(-1) # (batch_size*max_seq_len, )
    loss_per_token = self.loss_function(logits_flat, targets_flat) 
    loss_per_token *= (-1)  # nn.CELoss gives negative log likelihood but what we need is log<x_theta, x>
    loss_per_sample = loss_per_token.view(batch_size, seq_len).sum(dim=1)  # (batch_size, )  
    batchwise_loss = (loss_weight * loss_per_sample).mean() # Positive value
    # print(batchwise_loss.shape)

    return batchwise_loss

  def add_noise(self, x, t, attention_mask, noise_schedule):
    # Mask token with prob. 1-alpha_t
    alpha_t, loss_weight = self.get_weights(t, noise_schedule)
    alpha_t_unsqueezed = alpha_t.unsqueeze(1)  # (batch_size, 1)
    # print(f"alpha_t.shape : {alpha_t.shape}")

    masking_positions = (attention_mask == 1) & \
                        (x != self.tokenizer.cls_token_id) & (x != self.tokenizer.sep_token_id)
    # print(masking_positions)

    rand_tensor = torch.rand(x.shape, device=x.device, dtype=torch.float32)
    candidates = rand_tensor > alpha_t_unsqueezed  # Mask with 1-alpha_t
    diffusion_mask = masking_positions & candidates
    # print(f"diffusion_mask.shape : {diffusion_mask.shape}")

    noised_x = x.clone()
    noised_x[diffusion_mask] = self.tokenizer.mask_token_id

    return noised_x, diffusion_mask, loss_weight
  
  def get_weights(self, t, noise_schedule="cosine"):
    if noise_schedule == "cosine":
      alpha_t = torch.cos((math.pi/2) * t)
      alpha_t_prime = -torch.sin((math.pi/2) * t) * (math.pi/2) # Negative value
      alpha_t_prime = alpha_t_prime
    else:
      raise NotImplementedError(f"Noise schedule {noise_schedule} not implemented.")
    
    one_minus_alpha_t = 1 - alpha_t
    one_minus_alpha_t = torch.clamp(one_minus_alpha_t, min=1e-9)  # Avoid zero-division
    loss_weight = alpha_t_prime / one_minus_alpha_t # Negative value
    
    return alpha_t, loss_weight

  @torch.no_grad()
  def sample(self, num_samples, steps, device='cuda'):
    # Generate sequence(s) of mask tokens
    z_t = torch.full(
      (num_samples, self.model.max_seq_len),
      self.tokenizer.mask_token_id,   # Fill with [MASK] tokens
      dtype=torch.long, device=device
    )
    
    # Generate time step schedule
    time_steps = torch.linspace(1.0, 0.0, steps+1, device=device)
      
    # One-hot encoding of mask tokens (i.e. 1 only at mask_token_id index, 0 elsewhere)
    m_one_hot = F.one_hot(
      torch.tensor(self.tokenizer.mask_token_id, device=device),
      num_classes=self.tokenizer.vocab_size,
    ).to(dtype=torch.float32)  # (vocab_size, )
    m_one_hot = m_one_hot.view(1,1,-1)  # (1, 1, vocab_size)
    
    # Sampling loop
    for i in tqdm(range(steps)):
      # Distinguish masked and unmasked tokens
      is_mask = (z_t == self.tokenizer.mask_token_id)  # m (num_samples, max_seq_len)      
      if not is_mask.any():   # If no masked tokens remain, break loop.
        break      
      
      # Time Steps : 0 < s < t < 1
      t_curr_vec = torch.full((num_samples,), time_steps[i], device=device)   # t
      t_prev_vec = torch.full((num_samples,), time_steps[i+1], device=device) # s
      
      # Get the logits from the model and calculate proabilities
      x_theta_logit = self.model(z_t, t_curr_vec, attention_mask=None)  # (num_samples, max_seq_len, vocab_size)
      x_theta_probs = F.softmax(x_theta_logit, dim=-1)  # (num_samples, max_seq_len, vocab_size)
      
      # Get weights
      alpha_t, _ = self.get_weights(t_curr_vec, noise_schedule="cosine")  # (num_samples, )
      alpha_s, _ = self.get_weights(t_prev_vec, noise_schedule="cosine")  # (num_samples, )
      # Broadcast to match dim with z_t
      alpha_t, alpha_s = alpha_t.view(-1,1,1), alpha_s.view(-1,1,1)  # (num_samples, 1, 1)
      
      # Calculate the backward Categorical distribution for masked tokens
      term_m = (1-alpha_s) * m_one_hot  # (num_samples, 1, vocab_size)
      term_x_theta = (alpha_s - alpha_t) * x_theta_probs  # (num_samples, max_seq_len, vocab_size)
      denominator = torch.clamp(1-alpha_t, min=1e-9)  # Avoid zero-division
      prob_unmask = (term_m + term_x_theta) / denominator  # (num_samples, max_seq_len, vocab_size)
      
      # Flatten to index the masked tokens only
      is_mask_flat = is_mask.view(-1) # (num_samples*max_seq_len, )
      prob_unmask_flat = prob_unmask.view(-1, self.tokenizer.vocab_size)  # (num_samples*max_seq_len, vocab_size)
      prob_unmask_target = prob_unmask_flat[is_mask_flat]  # (<number of unmasked tokens>, vocab_size)
      
      # Sample from the Categorical distribution
      sample_tokens = torch.multinomial(
        prob_unmask_target,
        num_samples=1,
      ).squeeze(-1)  # (<number of unmasked tokens>, )
      
      
      # Create z_s : For already unmasked tokens, keep the same. Update only the masked tonkens.
      z_s = torch.clone(z_t)
      z_s_flat = z_s.view(-1)  # (num_samples*max_seq_len, )
      z_s_flat[is_mask_flat] = sample_tokens
      
      # Update z_t 
      z_t = z_s.view(num_samples, self.model.max_seq_len)  # (num_samples, max_seq_len)
    
    return z_t