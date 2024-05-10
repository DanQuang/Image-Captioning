import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import ImageCaptionModel
# from encoder import EncoderCNN
# from decoder import DecoderRNN
from dataset import Load_Data
import pandas as pd

class Test_Task:
    def __init__(self, config):
        self.save_path = config["save_path"]
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

        self.criterion = nn.CrossEntropyLoss(ignore_index= padding_idx)

    def predict(self):
        test_loader = self.dataloader.load_test()
        test_image = [data["image_path"] for data in self.dataloader.test]
        test_caption = [data["caption"] for data in self.dataloader.test]

        best_model = "ImageCaptionModel_best_model.pth"

        if os.path.join(self.save_path, best_model):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            print("Loading best model.....")
            self.model.load_state_dict(checkpoint["model_state_dict"])

            
            self.model.eval()
            with torch.inference_mode():
                epoch_loss = 0
                predict_tokens_list = []
                for _, item in enumerate(tqdm(test_loader)):
                    images, captions = item["image"].to(self.device), item["caption_ids"].to(self.device)

                    output = self.model(images, captions[:,:-1])

                    predict_token = output.argmax(-1)
                    predict_tokens_list.append(predict_token)

                    # outputs: [batch_size, seq_len, vocab_size]
                    output_dim = output.shape[-1] # target_vocab_size

                    output = output.contiguous().view(-1, output_dim)
                    # output: [batch_size*target_len, target_vocab_size]

                    target = captions.contiguous().view(-1)
                    # target: [batch_size*target_len]

                    loss = self.criterion(output, target)

                    epoch_loss += loss.item()

                test_loss = epoch_loss / len(test_loader)

                print(f"Test loss: {test_loss:.5f}")

                concatenated_tokens = torch.cat(predict_tokens_list, dim=0).tolist()

                list_sentence = [' '.join(self.dataloader.dataset.vocab.convert_ids_to_tokens(ids)) for ids in concatenated_tokens]

                # make csv file
                df = pd.DataFrame({"image": test_image,
                                   "caption": test_caption,
                                   "predict": list_sentence})
                df.to_csv("result.csv", index= False)