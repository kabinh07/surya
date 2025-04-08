import torch
import os
from surya.recognition.loader import RecognitionModelLoader
from surya.settings import Settings
from torch.utils.data import random_split, Dataset
import cv2
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
import pandas as pd
import numpy as np

settings=Settings()
def check_for_nan_inf(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"⚠ NaN or Inf found in {name}")
        raise ValueError("NaN or Inf in logits")

class CustomTrainer(Trainer):
    def _gen_output(self, model, inputs):
        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            encoder_model = model.module.encoder
            decoder_model = model.module.decoder
            text_encoder_model = model.module.text_encoder
            device=model.module.device
        else:
            encoder_model = model.encoder
            decoder_model = model.decoder
            text_encoder_model = model.text_encoder
            device=model.device

        # Forward pass through encoder
        pixels = inputs['pixel_values'].to(settings.MODEL_DTYPE)
        batch_langs = inputs["langs"]
        batch_pixel_values, batch_decoder_input = self.prepare_input(
            batch_langs,
            pixels,
            1
        )

        encoder_hidden_states = encoder_model(pixel_values=pixels).last_hidden_state
        # encoder_hidden_states.retain_grad()
        # token_count = 0
        # inference_token_count = batch_decoder_input.shape[-1]
        # decoder_position_ids = torch.ones_like(batch_decoder_input[0, :], dtype=torch.int64,
        #                                            device=self.model.device).cumsum(0) - 1
        
        # sequence_scores = torch.zeros(batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device).unsqueeze(1)
        # all_done = torch.zeros(batch_pixel_values.shape[0], dtype=torch.bool, device=self.model.device)
        # batch_predictions = torch.zeros(batch_pixel_values.shape[0], dtype=torch.int64, device=self.model.device).unsqueeze(1)
        # device_pad_token = torch.tensor(self.processor.tokenizer.pad_token_id, device=self.model.device)
        encoder_hidden_states = encoder_model(pixel_values=batch_pixel_values).last_hidden_state
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
                        input_ids=batch_decoder_input,
                        encoder_hidden_states=encoder_text_hidden_states,
                        
                    )
        return outputs,batch_decoder_input
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract sub-models
        
        # Prepare input for text encoder
        # text_encoder_input_ids = torch.arange(
        #     text_encoder_model.config.query_token_count,
        #     device=encoder_hidden_states.device,
        #     dtype=torch.long
        # ).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)

        # Forward pass through text encoder
        # encoder_text_hidden_states = text_encoder_model(
        #     input_ids=text_encoder_input_ids,
        #     cache_position=None,
        #     attention_mask=None,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=None,
        #     use_cache=False
        # ).hidden_states
        # encoder_text_hidden_states.retain_grad()
        
        # Forward pass through decoder
        # outputs = decoder_model(
        #     input_ids=inputs['decoder_input_ids'],
        #     encoder_hidden_states=encoder_text_hidden_states,
        # )
        outputs,batch_decoder_input = self._gen_output(model, inputs)
        
        logits = outputs["logits"]
        check_for_nan_inf(logits, "logits")
        check_for_nan_inf(batch_decoder_input, "decoder_input_ids")
        # Compute loss
        PAD_TOKEN_ID = 0  # Replace with your model's actual padding token ID
        logits = torch.clamp(logits, min=-10, max=10)
        # print("Logits stats → min:", logits.min().item(), "max:", logits.max().item(), "mean:", logits.mean().item())

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID,label_smoothing=0.1)
        # loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), batch_decoder_input.view(-1))
        outputs['batch_decoder_input'] = batch_decoder_input
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs,num_items_in_batch=None):
        """Override training step to add gradient clipping."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass & loss computation
        loss = self.compute_loss(model, inputs)
        with torch.autograd.set_detect_anomaly(True):
        
            # Backward pass
            loss.backward()

        # ✅ Apply gradient clipping here
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return loss
    def prepare_input(self, batch_langs, batch_pixel_values, batch_size):
        # print(batch_langs, "batch_langs")
        batch_decoder_input = [[self.model.config.decoder_start_token_id] + lang.tolist() for lang in batch_langs]
        max_input_length = max(len(tokens) for tokens in batch_decoder_input)
        
        # Pad decoder input to max length if needed, to ensure we can convert to a tensor
        for idx, tokens in enumerate(batch_decoder_input):
            if len(tokens) < max_input_length:
                padding_length = max_input_length - len(tokens)
                batch_decoder_input[idx] = [self.processor.tokenizer.pad_id] * padding_length + tokens

        # batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=self.model.dtype)
        batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long)
        

        # Moving this after the padding fixes XLA recompilation issues
        batch_pixel_values = batch_pixel_values.to(self.model.device)
        batch_decoder_input = batch_decoder_input.to(self.model.device)

        return batch_pixel_values, batch_decoder_input
    
    def prediction_step(self, model, inputs, prediction_loss_only=False,ignore_keys=None):
        """Custom evaluation step."""
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)

            # Forward pass
            # outputs = model(**inputs)
            # logits = outputs["logits"]

            # Compute loss if required
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return loss,None,None

            # Custom evaluation logic (e.g., accuracy, metrics)
            # outputs, batch_decoder_input = self._gen_output(model, inputs)
            # logits = outputs["logits"]
            # predictions = torch.argmax(logits, dim=-1)
            # labels = batch_decoder_input
            # Check for NaN or Inf in logits    

            # Example: Compute accuracy
            loss,outputs = self._gen_output(model, inputs)
            logits = outputs["logits"]
            labels = outputs["batch_decoder_input"]
            # predictions = torch.argmax(logits[:, -1], dim=-1)
            # accuracy = (predictions == labels).float().mean().item()

            return  loss,  logits, labels

class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        # Ensure all necessary keys exist before calling super()
        for feature in features:
            if "input_ids" not in feature:
                raise KeyError("'input_ids' is missing from the batch, cannot create 'decoder_input_ids'.")

        # Add decoder-specific keys before padding
        for feature in features:
            feature['decoder_input_ids'] = feature["input_ids"]
            feature['decoder_attention_mask'] = feature.get("attention_mask", None)

        # Apply padding only (remove truncation)
        batch = self.tokenizer.pad(
            features,
            padding=True,  # Ensures consistent sequence length
            return_tensors="pt"  # Ensures tensors are returned
        )

        # print("Batch keys after modification:", batch.keys())
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
        embeddings = self.processor(
            images=image,
            text=text,
            langs=['en','bn'],
            return_tensors="pt",
            padding="max_length",  # Ensures max length padding
            truncation=True,  # Ensures truncation if sequences are too long
            max_length=100  # Adjust this based on your model's expected input length
        )

        return {
            "pixel_values": embeddings['pixel_values'].squeeze(0).to(settings.MODEL_DTYPE),
            "input_ids": torch.tensor(embeddings['labels'], dtype=torch.long).squeeze(0),
            "langs": embeddings["langs"],
            "attention_mask": embeddings["attention_mask"]
        }

if __name__ == "__main__":
    recognition_model = RecognitionModelLoader("./recognition_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = recognition_model.processor()
    model = recognition_model.model()
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    collate_fn = CustomDataCollatorWithPadding(tokenizer=processor.tokenizer, pad_to_multiple_of=8, return_tensors = 'pt')
    df = pd.read_csv(os.path.join(os.path.abspath(''), "data/labels.csv"))
    # df=df[:20000]
    # Define split sizes
    train_size = int(0.8 * len(df))  # 80% training
    val_size =  len(df)-train_size    # 20% validation
    df.reset_index(inplace = True, drop = True)
    data = OCRDataset(dataframe=df, processor=processor)
    train_data,test_data=random_split(data,[train_size,val_size])
    print(f"TRAINING DATA LENGTH LENGHT: {len(train_data)}\nTEST DATA LENGTH: {len(test_data)}")
    

    training_args = TrainingArguments(
        output_dir="./experiment",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=10000,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=100,
        save_total_limit=1,
        fp16=False,  # Enable mixed precision training
        report_to="tensorboard",
        warmup_steps=500,
        warmup_ratio=0.03,
        gradient_accumulation_steps=2,
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=1.0,  # Add gradient clipping
        # bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    )
    ignore_keys_for_eval=["labels","input_ids"]
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        preprocess_logits_for_metrics=None,
    )
    trainer.train(
    ignore_keys_for_eval=ignore_keys_for_eval,
    )
    # torchrun --nproc-per-node=4 --master-addr="localhost" --master-port=12355 train.py%
