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
    self.loss_function = nn.CrossEntropyLoss(
      ignore_index=self.tokenizer.pad_token_id,
      reduction='none'  # To weight with alpha_t_prime / (1-alpha_t)
    )

  def forward(self, x, attention_mask):
    batch_size, seq_len = x.shape

    # Sample time
    t = torch.rand((batch_size,)).to(x.device)

    # Draw latent z_t from the Categorical distribution and update attention_mask
    z_t, diffusion_mask, alpha_t = self.add_noise(x, t, attention_mask, self._cosine_schedule)
    
    # Get loss weight : alpha_t_prime / (1-alpha_t)
    t.requires_grad_(True)
    alpha_t_prime = torch.autograd.grad(
      outputs=alpha_t,
      inputs=t,
      grad_outputs=torch.ones_like(alpha_t, device=x.device),
      create_graph=False,
      retain_graph=False,
    )[0]
    t.requires_grad_(False)
    one_minus_alpha_t = 1 - alpha_t.detach()
    one_minus_alpha_t = torch.clamp(one_minus_alpha_t, min=1e-9)  # Avoid zero-division
    loss_weight = alpha_t_prime.detach() / one_minus_alpha_t

    # Get x_theta
    x_theta = self.model(z_t, t, attention_mask)
    # print(f"x.shape : {x.shape}, z_t.shape : {z_t.shape}, x_theta.shape : {x_theta.shape}")

    # Get Cross Entropy Loss
    logits_flat = x_theta.view(-1, self.tokenizer.vocab_size)  # (batch_size*max_seq_len, vocab_size)
    targets = x.clone()
    targets[~diffusion_mask] = self.tokenizer.pad_token_id
    targets_flat = targets.view(-1) # (batch_size*max_seq_len, )
    loss_per_token = self.loss_function(logits_flat, targets_flat)
    loss_per_sample = loss_per_token.view(batch_size, seq_len).sum(dim=1)  # (batch_size, )    

    return (loss_weight * loss_per_sample).mean()

  def _cosine_schedule(self, t):
    # assert 0 <= t <= 1
    # if t == 1:  # cos(0) -> -log(1) -> 0
    #   return math.inf
    # elif t == 0:  # cos(2/pi) -> -log(0) -> inf
    #   return 0
    return -torch.log(torch.cos(math.pi/2 * (t)))

  def add_noise(self, x, t, attention_mask, noise_schedule):
    # Mask token with prob. 1-alpha_t
    alpha_t = torch.exp(-noise_schedule(t)).to(x.device)  # (batch_size, )
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

    return noised_x, diffusion_mask, alpha_t

  @torch.no_grad()
  def sample(self, z):
    pass