# augmented_image_buffer.py
import random
import torch

class AugmentedImageBuffer:
    """
    Memorize finals augmented images and their labels.
    ! This is NOT the Replay buffer !
    Attributes:
        capacity (int): buffer dimension.
        buffer (List): list of tuples (image_tensor, true_label).
        ptr (int): pointer to find oldest elements.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.ptr = 0

    def add(self, image_tensor, true_label):
        """
        Add a new tuple (image, label) to the buffer.
        """

        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # Add placeholder if max capacity is not reached
        
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
            
        self.buffer[self.ptr] = (image_tensor.clone().detach(), true_label)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        """
        Extract a sample of dimension batch_size from the buffer.
        """
        
        if len(self.buffer) < batch_size:
            return None, None # Not enough images in the buffer
        
        batch = random.sample(self.buffer, batch_size)
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
    
    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.ptr = 0