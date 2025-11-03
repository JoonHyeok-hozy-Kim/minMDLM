import torch
import torch.nn as nn
import math

class MDLM(nn.Module):
  def __init__(
      self, 
      model,
      tokenizer,
  ):
    super().__init__()
    self.model = model
    self.tokenizer = tokenizer
    self.loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

  def forward(self, x, attention_mask):
    batch_size, seq_len = x.shape

    # Sample time
    t = torch.rand((batch_size,)).to(x.device)

    # Draw latent z_t from the Categorical distribution and update attention_mask
    z_t, diffusion_mask = self.add_noise(x, t, attention_mask, self._cosine_schedule)

    # Get x_theta
    x_theta = self.model(z_t, t, attention_mask)
    # print(f"x.shape : {x.shape}, z_t.shape : {z_t.shape}, x_theta.shape : {x_theta.shape}")

    # Get Loss
    logits_flat = x_theta.view(-1, self.tokenizer.vocab_size)  # (batch_size*max_seq_len, vocab_size)
    targets = x.clone()
    targets[~diffusion_mask] = self.tokenizer.pad_token_id
    targets_flat = targets.view(-1) # (batch_size*max_seq_len, )
    batchwise_ce_loss = self.loss_function(logits_flat, targets_flat)

    return batchwise_ce_loss

  def _cosine_schedule(self, t):
    # assert 0 <= t <= 1
    # if t == 1:  # cos(0) -> -log(1) -> 0
    #   return math.inf
    # elif t == 0:  # cos(2/pi) -> -log(0) -> inf
    #   return 0
    return -torch.log(torch.cos(math.pi/2 * (t)))

  def add_noise(self, x, t, attention_mask, noise_schedule):
    # Mask token with prob. 1-alpha_t
    alpha_t = torch.exp(-noise_schedule(t))
    alpha_t = alpha_t.unsqueeze(1)  # To match the shape of rand_tensor
    # print(f"alpha_t.shape : {alpha_t.shape}")

    masking_positions = (attention_mask == 1) & \
                        (x != self.tokenizer.cls_token_id) & (x != self.tokenizer.sep_token_id)
    # print(masking_positions)

    rand_tensor = torch.rand(x.shape).to(x.device)
    candidates = rand_tensor > alpha_t  # Mask with 1-alpha_t
    diffusion_mask = masking_positions & candidates
    # print(f"diffusion_mask.shape : {diffusion_mask.shape}")

    noised_x = x.clone()
    noised_x[diffusion_mask] = self.tokenizer.mask_token_id

    return noised_x, diffusion_mask

  @torch.no_grad()
  def sample(self, z):
    pass