import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import (
    ViTFeatureExtractor,
    BartTokenizer,
    VisionEncoderDecoderModel
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from tqdm import tqdm

os.makedirs('data/images/train', exist_ok=True)
os.makedirs('data/images/val', exist_ok=True)

dummy_captions = [
    {"image_id": "1", "caption": "A cat sitting on a window sill."},
    {"image_id": "2", "caption": "A group of people standing around a car."},
    {"image_id": "3", "caption": "A beautiful sunset over the mountains."}
]

with open('data/captions_train.json', 'w') as f:
    json.dump(dummy_captions, f)

for i in range(1, 4):
    img = Image.new('RGB', (224, 224), color=(i*40, i*40, i*40))
    img.save(f'data/images/train/{i}.jpg')

class COCODataset(Dataset):
    def __init__(self, captions_file, images_dir, feature_extractor, tokenizer, max_length=30):
        self.captions = json.load(open(captions_file, 'r'))
        self.images_dir = images_dir
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        image_id = self.captions[idx]['image_id']
        image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze()
        tokens = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze()
        attention_mask = tokens['attention_mask'].squeeze()
        return pixel_values, input_ids, attention_mask

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

train_dataset = COCODataset(
    captions_file='data/captions_train.json',
    images_dir='data/images/train',
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    max_length=30
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    "google/vit-base-patch16-224",
    "facebook/bart-base"
)

for param in model.encoder.parameters():
    param.requires_grad = False

model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.to('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, captions, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        captions = captions.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            pixel_values=images,
            labels=captions,
            decoder_attention_mask=masks
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

model.eval()

def generate_caption(image_path, model, tokenizer, feature_extractor, device, max_length=30):
    image = Image.open(image_path).convert('RGB')
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

dummy_image_path = 'arbitraryImagePath' # replace and insert file path here
caption = generate_caption(dummy_image_path, model, tokenizer, feature_extractor, device)
print(f"Generated Caption: {caption}")

torch.save(model.state_dict(), 'VLMmodel.pth')
print("Model saved successfully.")