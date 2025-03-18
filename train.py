import torch
import os
from surya.recognition.loader import RecognitionModelLoader
from torch.utils.data import Dataset
import cv2
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import pandas as pd
from typing import Dict, Union, Any

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        encoder_model = None
        decoder_model = None
        text_encoder_model = None
        if isinstance(model, torch.nn.DataParallel):
            encoder_model = model.module.encoder
            decoder_model = model.module.decoder
            text_encoder_model = model.module.text_encoder
        else:
            encoder_model = model.encoder
            decoder_model = model.decoder
            text_encoder_model = model.text_encoder
        pixels = inputs['pixel_values'].to(torch.float16)
        encoder_hidden_states = encoder_model(pixel_values=pixels).last_hidden_state
        text_encoder_input_ids = torch.arange(
                    text_encoder_model.config.query_token_count,
                    device=encoder_hidden_states.device,
                    dtype=torch.long
                ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)
        encoder_text_hidden_states = text_encoder_model(
                    input_ids=text_encoder_input_ids,
                    cache_position=None,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    use_cache=False
                ).hidden_states
        outputs = decoder_model(
                        input_ids=inputs['decoder_input_ids'],
                        encoder_hidden_states=encoder_text_hidden_states,
                    )
        logits = outputs["logits"]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), inputs['decoder_input_ids'].view(-1))

        return (loss, outputs) if return_outputs else loss

class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        batch['decoder_input_ids'] = batch.pop('input_ids')
        batch['decoder_attention_mask'] = batch.pop('attention_mask')
        return batch

class OCRDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.images = dataframe['filename'].to_list()
        self.labels = dataframe['words'].to_list()
        self.langs = dataframe['langs'].to_list()
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(os.path.abspath(''), "data/images", self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text = self.labels[idx]
        lang = self.langs[idx]
        embeddings = self.processor(
            images= image,
            text= text,
            langs= [lang],
            return_tensors = "pt"
        )
        return {
            "pixel_values": embeddings['pixel_values'].squeeze(0).to(torch.float16),
            "input_ids": torch.tensor(embeddings['labels'], dtype=torch.long),
        }

if __name__ == "__main__":
    recognition_model = RecognitionModelLoader("./recognition_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = recognition_model.processor()
    model = recognition_model.model()
    collate_fn = CustomDataCollatorWithPadding(tokenizer=processor.tokenizer, pad_to_multiple_of=8, return_tensors = 'pt')
    df = pd.read_csv(os.path.join(os.path.abspath(''), "data/labels.csv"))
    train_df = df.iloc[:6]
    test_df = df.iloc[6:-1]
    train_df.reset_index(inplace = True, drop = True)
    test_df.reset_index(inplace = True, drop = True)
    train_data = OCRDataset(dataframe=train_df, processor=processor)
    test_data = OCRDataset(dataframe=test_df, processor=processor)

    training_args = TrainingArguments(
        output_dir="./experiment",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=3,
        save_total_limit=3,
        fp16=True,
        ddp_find_unused_parameters=True,
        local_rank=-1,
        remove_unused_columns=False,
        report_to="tensorboard",
    )

    trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    )

    trainer.train()
    # torchrun --nproc-per-node=4 --master-addr="localhost" --master-port=12355 train.py%   
