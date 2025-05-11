from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import torch
import numpy as np
from PIL import Image
import base64
import io

class RecognizerModel(torch.nn.Module):
    def __init__(self, encoder, text_encoder, decoder, query_token_count: int, decoder_start_token_id: int, eos_token_id: int, max_len: int = 25):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder

        self.query_token_count = query_token_count
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len

    def forward(self, pixel_values):
        # 1. Vision Encoder
        with torch.no_grad():
            encoder_outputs = self.encoder(pixel_values=pixel_values)
            encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, N, D]

            batch_size = encoder_hidden_states.size(0)
            query_ids = torch.arange(self.query_token_count, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)

            # 2. Text Encoder
            text_encoder_outputs = self.text_encoder(
                input_ids=query_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=torch.ones(encoder_hidden_states.shape[:2], device=pixel_values.device),
                return_dict=False,
            )
            text_hidden_states = text_encoder_outputs[0]

            # 3. Decoder - fixed loop for tracing compatibility
            decoder_input_ids = torch.full(
                (batch_size, 1),
                self.decoder_start_token_id,
                dtype=torch.long,
                device=pixel_values.device,
            )

            outputs = []
            for _ in range(25):
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=text_hidden_states,
                    encoder_attention_mask=torch.ones(text_hidden_states.shape[:2], device=pixel_values.device),
                    return_dict=False,
                )
                logits = decoder_outputs[0]
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                outputs.append(next_token)

        return decoder_input_ids[:, 1:]
    
class DetectionModel(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor):
        return self.detector(image).logits
    
if __name__ == '__main__':
    recognizer = RecognitionPredictor()
    detector = DetectionPredictor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detection Model
    image = Image.open("test_data/father_name_dummy.png").convert("RGB")
    image = image.resize((1200, 1200))
    model = DetectionModel(detector=detector.model)
    model = model.to(device)
    model.eval()
    processed_image = torch.tensor(detector.processor(np.array(image)).pixel_values).to(device)
    logits = model(processed_image)
    traced_model = torch.jit.trace(model, processed_image)
    traced_model.save("models/detector.pt")

    # correct_shape = [1200, 1200]
    # current_shape = list(logits.shape[2:])
    # if current_shape != correct_shape:
    #     logits = torch.nn.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

    # logits = logits.to(torch.float32).cpu().numpy()
    # traced_model = torch.jit.trace(model)



    ## Recognizer Model
    # encoder = recognizer.model.encoder
    # text_encoder = recognizer.model.text_encoder
    # decoder = recognizer.model.decoder
    # query_token_count = recognizer.model.text_encoder.config.query_token_count
    # decoder_start_token_id = recognizer.model.config.decoder_start_token_id
    # eos_token_id = recognizer.model.config.eos_token_id

    # model = RecognizerModel(
    #     encoder=encoder,
    #     decoder=decoder,
    #     text_encoder=text_encoder,
    #     query_token_count=query_token_count,
    #     decoder_start_token_id=decoder_start_token_id,
    #     eos_token_id=eos_token_id
    # )

    # model = torch.jit.load("models/traced_model.pt")

    # image = Image.open("test_data/father_name_dummy.png").convert('RGB')
    # pixel_values = recognizer.processor.image_processor(image).pixel_values
    # model = model.to(device)
    # pixel_values = torch.tensor(pixel_values, dtype=torch.float32, device=device)  # Ensure correct dtype and device
    # output = model(pixel_values)
    # detected_text = recognizer.processor.tokenizer.batch_decode(output.cpu())
    # print(detected_text)
    # # dummy_input = pixel_values[0].unsqueeze(0)
    # # traced_model = torch.jit.trace(model, dummy_input)
    # # traced_model.save("models/traced_model.pt")
    # # exported_model: torch.export.ExportedProgram = torch.export.export(model, (dummy_input,))
    # # print("model exported...")
    # # torch.export.save(exported_model, "models/recognizer_exported.pt")

