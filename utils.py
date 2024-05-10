import re
import torch

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def padding(array, max_length, padding_value):
    """
    Input:
        array: list of words in the sequence that need padding
        max_length: output of word list after padding
        padding_value: padding value
    Output:
        a tensor of the sequence after padding
    """
    if len(array) < max_length:
            padding_length = max_length - len(array)
            padding_array = array + [padding_value]*padding_length
            return padding_array
    else:
        return array[:max_length]
    
def collate_fn(batch):
    batch_image = [data["image"] for data in batch]
    batch_caption_ids = [data["caption_ids"] for data in batch]

    return {
         "image": torch.tensor(batch_image),
         "caption_ids": torch.tensor(batch_caption_ids)
    }