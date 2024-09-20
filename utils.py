import torch
import torchvision
from torch import nn
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.functional.regression import mean_squared_error

from sklearn.kernel_approximation import RBFSampler

import os
from einops import rearrange

import matplotlib.pyplot as plt


# function to mask the image
def mask_image(img,
               prop,
               device):
    img_copy = img.clone().to(device)
    mask = torch.rand(img.shape[1:]) < prop # uniform distribution
    img_copy[0][mask] = float('nan')
    img_copy[1][mask] = float('nan')
    img_copy[2][mask] = float('nan')
    return img_copy, mask

# function that randomly removes 900 pixels from the image
def random_mask_image(img, device):
    img_copy = img.clone().to(device)
    h, w = img.shape[1], img.shape[2]
    total_pixels = h * w
    
    # Randomly select 900 pixel indices
    random_indices = torch.randperm(total_pixels)[:900]
    
    # Convert flat indices back to 2D indices (for height and width)
    mask = torch.zeros(h, w, dtype=torch.bool, device=device)
    mask.view(-1)[random_indices] = True
    
    # Apply NaN mask to each channel
    img_copy[0][mask] = float('nan')
    img_copy[1][mask] = float('nan')
    img_copy[2][mask] = float('nan')
    
    return img_copy, mask

# function to select a random patch from image
def random_patch(img, patch_size):
  # Get the dimensions of the image
  height, width = img.shape[1:]

  # Calculate the maximum possible starting positions for the patch
  max_x = width - patch_size
  max_y = height - patch_size

  # Generate random starting positions within the bounds
  start_x = torch.randint(0, max_x + 1, (1,))
  start_y = torch.randint(0, max_y + 1, (1,))

  # Extract the patch from the image
  patch = img[:, start_y:start_y + patch_size, start_x:start_x + patch_size]

  return patch, start_x, start_y

# function to perform matrix factorization
def factorize(A,
              k,
              device):
    """Factorize the matrix D into A and B"""
    A = A.to(device)
    # Randomly initialize A and B
    W = torch.randn(A.shape[0], k, requires_grad=True, device=device)
    H = torch.randn(k, A.shape[1], requires_grad=True, device=device)
    # Optimizer
    optimizer = torch.optim.Adam([W, H], lr=0.01)
    mask = ~torch.isnan(A)
    # Train the model
    for i in range(1000):
        # Compute the loss
        diff_matrix = torch.mm(W, H) - A
        diff_vector = diff_matrix[mask] # makes a 1D tensor
        loss = torch.norm(diff_vector)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Backpropagate
        loss.backward()
        
        # Update the parameters
        optimizer.step()
        
    return W, H, loss

# function to compute the RMSE and PSNR
def metrics(img, reconstructed_img):
    rmse = mean_squared_error(target = img.reshape(-1),
                             preds=reconstructed_img.reshape(-1),
                             squared=False)
    psnr = peak_signal_noise_ratio(target=img.reshape(-1),
                                   preds=reconstructed_img.reshape(-1))
    return rmse, psnr

############################################################

def load_image(path):
  if (os.path.exists(path)):
    # read the image using torchvision and convert it to tensor because the model expects a tensor
    img = torchvision.io.read_image(path)
    print("Tensor shape: ", img.shape)
    # rearrange the image to be in the expected format i.e. (H W C)
    plt.imshow(rearrange(img, 'c h w -> h w c').numpy())
    return img
  else:
    print("File not found")
    return None


def normalize_image(image):
    # Convert the image to a float tensor if it's not already
    image = image.float()
    
    # Find the minimum and maximum pixel values
    min_val = torch.min(image)
    max_val = torch.max(image)
    
    # Normalize the image to the range [0, 1]
    scaled_image = (image - min_val) / (max_val - min_val)
    
    return scaled_image

def normalize_tensor(tensor, device='cpu'):
    tensor = tensor.to(device)

    # Find the min and max values of the tensor
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    # Normalize the tensor to [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    # Scale the tensor to the range [-1, 1]
    scaled_tensor = normalized_tensor * 2 - 1

    return scaled_tensor

def crop_image(image, crop_size, start_x, start_y): # crop a given image tensor with specified crop size and x, y start values
  cropped_image = torchvision.transforms.functional.crop(image.cpu(), start_x, start_y, crop_size, crop_size) # x.cpu() moves the tensor from gpu to cpu
  print("Cropped image shape: ", cropped_image.shape)
  plt.imshow(rearrange(cropped_image, 'c h w -> h w c').cpu().numpy())
  return cropped_image

def extract_coordinates_pixels(image, device): # this function extracts the coordinates and pixel values of an image tensor and returns them
  channels, height, width = image.shape
  coords = [] # store the coordinates

  for y in range(height):
    for x in range(width):
      coords.append([x, y])
      
  coords = torch.tensor(coords, dtype=torch.float32)
  pixel_values = rearrange(image, 'c h w -> (h w) c').float() # rearrange the image tensor to have the pixel values in the first dimension by flatten the height and width dimensions

  print("Coordinates shape: ", coords.shape)
  print("Pixel values shape: ", pixel_values.shape)
  return coords.to(device), pixel_values.to(device)

def extract_not_nan_coordinates_pixels(image, device): 
  channels, height, width = image.shape
  coords = []
  pixel_values = [] 

  for y in range(height):
    for x in range(width):
      if image[0][x][y].isnan():
        continue
      coords.append([x, y])
      pixel_values.append(image[:, x, y].tolist())

  coords = torch.tensor(coords, dtype=torch.float32)
  pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

  return coords.to(device), pixel_values.to(device)

def downsample_image(cropped_image, scale=2, device='cpu'):
    # Get the number of channels, original height, and width
    num_channels, height, width = cropped_image.shape

    # Compute new height and width
    new_height = height // scale
    new_width = width // scale

    # Create an empty tensor to store the downsampled image
    low_res_image = torch.zeros((num_channels, new_height, new_width)).to(device)

    # Perform average pooling over the 2x2 neighborhoods
    for i in range(new_height):
        for j in range(new_width):
            low_res_image[:, i, j] = torch.mean(cropped_image[:, scale * i:scale * i + scale, scale * j:scale * j + scale], dim=(1, 2))

    # Display the downsampled image
    plt.imshow(rearrange(low_res_image, 'c h w -> h w c').cpu().numpy())
    plt.show()

    # Return the downsampled image
    return low_res_image

# Example usage
# downsampled_img = downsample_image(cropped_image, scale=2, device='cuda')

def create_linear_model(input_dim, output_dim, device):
  return nn.Linear(input_dim, output_dim).to(device)

def train(coords, pixels, model, learning_rate=0.01, epochs=1000, threshold=1e-6, verbose=True):

    criterion = nn.MSELoss() # define the loss function (mse)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # use the adam optimizer with the specified learning rate
    previous_loss = float('inf') # initialize w very large value (for early stopping)

    # training loops
    for epoch in range(epochs):
        optimizer.zero_grad() # reset the gradient of the optimizer
        outputs = model(coords) # compute the output
        loss = criterion(outputs, pixels) # calculate the loss that we defined earlier
        loss.backward() # compute teh gradients of the loss with respect to the parameters
        optimizer.step() # update the parameters based on the gradients computed above

        # check for early stopping
        if abs(previous_loss - loss.item()) < threshold:
            print(f"Stopping early at epoch {epoch} with loss: {loss.item():.6f}")
            break

        previous_loss = loss.item() # update the previous loss

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")

    return loss.item()

def create_rff_features(tensor, num_features, sigma, device):
    rff = RBFSampler(n_components=num_features, gamma=1/(2 * sigma**2), random_state=42)
    tensor = torch.tensor(rff.fit_transform(tensor.cpu().numpy())).float().to(device)
    return tensor

def create_coordinate_map(height, width, device, inv=False): # given the height and width of an image this function creates a coordinate map and returns it
  coords = []
  if not inv:
    for x in range(height):
      for y in range(width):
        coords.append([x, y])
  else:
     for y in range(height):
      for x in range(width):
        coords.append([x, y])

  return torch.tensor(coords, dtype=torch.float32).to(device)

def get_reconstructed_image(model, coords, image_rff, height, width): # given a model, coordinates, RFF features, height, and width this function returns the reconstructed image
  model.eval()
  with torch.no_grad():
    outputs = model(image_rff)
    outputs = outputs.reshape(height, width, 3)
  return outputs