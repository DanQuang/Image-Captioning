import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import ImageCaptionModel
# from encoder import EncoderCNN
# from decoder import DecoderRNN
from dataset import Load_Data

class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        # self.dropout = config["dropout"]
        # self.batch_size = config["batch_size"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load data
        self.dataloader = Load_Data(config)

        # vocab size and padding_idx
        vocab_size = self.dataloader.dataset.vocab.vocab_size()
        padding_idx = self.dataloader.dataset.vocab.pad_idx()

        # Load model
        self.model = ImageCaptionModel(config, vocab_size, padding_idx).to(self.device)

        self.optim = optim.Adam(self.model.parameters(), lr= self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index= padding_idx)

    def train(self):
        train, dev = self.dataloader.load_train_dev()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = "ImageCaptionModel_last_model.pth"
        best_model = "ImageCaptionModel_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
                checkpoint = torch.load(os.path.join(self.save_path, last_model))
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optim.load_state_dict(checkpoint["optim_state_dict"])
                print("Load the last model")
                initial_epoch = checkpoint["epoch"] + 1
                print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0

        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            epoch_loss = 0
            for _, item in enumerate(tqdm(train)):
                images, captions = item["image"].to(self.device), item["caption_ids"].to(self.device)

                self.optim.zero_grad()

                output = self.model(images, captions[:,:-1])
                # outputs: [batch_size, seq_len, vocab_size]
                output_dim = output.shape[-1] # target_vocab_size

                output = output.contiguous().view(-1, output_dim)
                # output: [batch_size*target_len, target_vocab_size]

                target = captions.contiguous().view(-1)
                # target: [batch_size*target_len]

                loss = self.criterion(output, target)

                loss.backward()

                self.optim.step()

                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train)
            print(f"Epoch {epoch}:")
            print(f"Train loss: {train_loss:.5f}")

            epoch_loss = 0
            with torch.inference_mode():
                for _, item in enumerate(tqdm(dev)):
                    images, captions = item["image"].to(self.device), item["caption_ids"].to(self.device)

                    self.optim.zero_grad()

                    output = self.model(images, captions[:,:-1])
                    # outputs: [batch_size, seq_len, vocab_size]
                    output_dim = output.shape[-1] # target_vocab_size

                    output = output.contiguous().view(-1, output_dim)
                    # output: [batch_size*target_len, target_vocab_size]

                    target = captions.contiguous().view(-1)
                    # target: [batch_size*target_len]

                    loss = self.criterion(output, target)

                    epoch_loss += loss.item()

                valid_loss = epoch_loss / len(dev)

                print(f"Valid loss: {valid_loss:.5f}")

                score = valid_loss

                # save last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'score': score
                }, os.path.join(self.save_path, last_model))

                # save the best model
                if epoch > 0 and score > best_score:
                    threshold += 1
                else:
                    threshold = 0

                if score <= best_score or epoch == 0:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'score':score
                    }, os.path.join(self.save_path, best_model))
                    print(f"Saved the best model with valid loss of {score:.5f}")
