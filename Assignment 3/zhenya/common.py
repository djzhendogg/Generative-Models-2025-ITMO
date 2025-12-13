def reset_grad():
    ## reset gradient for optimizer of generator and discrimator
    g_optimizer.zero_grad()
    d_optimizer.zero_grad()


LABEL_SMOOTH = 0.95


def train_discriminator(images):
    real_smooth_max = 1.2
    real_smooth_min = 0.7
    fake_smooth_max = 0.3
    fake_smooth_min = 0.0
    FLIP_PROB = 0.05

    batch_size = images.size(0)
    real_labels_base = torch.rand(batch_size, 1).to(device) * (real_smooth_max - real_smooth_min) + real_smooth_min
    fake_labels_base = torch.rand(batch_size, 1).to(device) * (fake_smooth_max - fake_smooth_min) + fake_smooth_min

    flip_mask_real = (torch.rand(batch_size, 1).to(device) < FLIP_PROB)
    flipped_real_values = torch.rand(batch_size, 1).to(device) * (fake_smooth_max - fake_smooth_min) + fake_smooth_min
    real_labels = torch.where(flip_mask_real, flipped_real_values, real_labels_base)

    flip_mask_fake = (torch.rand(batch_size, 1).to(device) < FLIP_PROB)
    flipped_fake_values = torch.rand(batch_size, 1).to(device) * (real_smooth_max - real_smooth_min) + real_smooth_min
    fake_labels = torch.where(flip_mask_fake, flipped_fake_values, fake_labels_base)

    real_labels = torch.ones(images.size(0), 1).to(device) * LABEL_SMOOTH
    fake_labels = torch.ones(images.size(0), 1).to(device) * (1 - LABEL_SMOOTH)
    images = images.view(images.size(0), CFG.nc, CFG.image_size, CFG.image_size).to(device)

    outputs = D(images)
    # Loss for real images
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # Loss for fake images

    z = torch.randn(batch_size, CFG.nz, 1, 1).to(device)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # Sum losses
    d_loss = d_loss_real + d_loss_fake

    # Reset gradients
    reset_grad()

    d_loss.backward()
    d_optimizer.step()

    return d_loss, real_score, fake_score