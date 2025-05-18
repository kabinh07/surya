from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.recognition.loader import RecognitionModelLoader
import torch
import numpy as np
from PIL import Image
import base64
import io

from typing import Dict, Union, Optional, List, Iterable

import cv2
from torch import TensorType
from transformers import DonutImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_transforms import pad, normalize
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, get_image_size
import numpy as np
from PIL import Image

from transformers import ByT5Tokenizer

import PIL

# Tokenizer
TOTAL_TOKENS = 65536
TOKEN_OFFSET = 3 # Pad, eos, bos
SPECIAL_TOKENS = 253
TOTAL_VOCAB_SIZE = TOTAL_TOKENS + TOKEN_OFFSET + SPECIAL_TOKENS
LANGUAGE_MAP = {
    'af': 0,
    'am': 1,
    'ar': 2,
    'as': 3,
    'az': 4,
    'be': 5,
    'bg': 6,
    'bn': 7,
    'br': 8,
    'bs': 9,
    'ca': 10,
    'cs': 11,
    'cy': 12,
    'da': 13,
    'de': 14,
    'el': 15,
    'en': 16,
    'eo': 17,
    'es': 18,
    'et': 19,
    'eu': 20,
    'fa': 21,
    'fi': 22,
    'fr': 23,
    'fy': 24,
    'ga': 25,
    'gd': 26,
    'gl': 27,
    'gu': 28,
    'ha': 29,
    'he': 30,
    'hi': 31,
    'hr': 32,
    'hu': 33,
    'hy': 34,
    'id': 35,
    'is': 36,
    'it': 37,
    'ja': 38,
    'jv': 39,
    'ka': 40,
    'kk': 41,
    'km': 42,
    'kn': 43,
    'ko': 44,
    'ku': 45,
    'ky': 46,
    'la': 47,
    'lo': 48,
    'lt': 49,
    'lv': 50,
    'mg': 51,
    'mk': 52,
    'ml': 53,
    'mn': 54,
    'mr': 55,
    'ms': 56,
    'my': 57,
    'ne': 58,
    'nl': 59,
    'no': 60,
    'om': 61,
    'or': 62,
    'pa': 63,
    'pl': 64,
    'ps': 65,
    'pt': 66,
    'ro': 67,
    'ru': 68,
    'sa': 69,
    'sd': 70,
    'si': 71,
    'sk': 72,
    'sl': 73,
    'so': 74,
    'sq': 75,
    'sr': 76,
    'su': 77,
    'sv': 78,
    'sw': 79,
    'ta': 80,
    'te': 81,
    'th': 82,
    'tl': 83,
    'tr': 84,
    'ug': 85,
    'uk': 86,
    'ur': 87,
    'uz': 88,
    'vi': 89,
    'xh': 90,
    'yi': 91,
    'zh': 92,
    "_math": 93
}

def text_to_utf16_numbers(text):
    utf16_bytes = text.encode('utf-16le')  # Little-endian to simplify byte order handling

    numbers = []

    # Iterate through each pair of bytes and combine them into a single number
    for i in range(0, len(utf16_bytes), 2):
        # Combine two adjacent bytes into a single number
        number = utf16_bytes[i] + (utf16_bytes[i + 1] << 8)
        numbers.append(number)

    return numbers


def utf16_numbers_to_text(numbers):
    byte_array = bytearray()
    for number in numbers:
        # Extract the two bytes from the number and add them to the byte array
        byte_array.append(number & 0xFF)         # Lower byte
        byte_array.append((number >> 8) & 0xFF)  # Upper byte

    try:
        text = byte_array.decode('utf-16le', errors="ignore")
    except Exception as e:
        print(f"Error decoding utf16: {e}")
        text = ""

    return text


def _tokenize(text: str, langs: List[str] | None, eos_token_id: int = 1, add_eos: bool = False, add_bos: bool = True):
    tokens = text_to_utf16_numbers(text)
    tokens = [t + TOKEN_OFFSET for t in tokens] # Account for special pad, etc, tokens

    lang_list = []
    if langs:
        for lang in langs:
            code = LANGUAGE_MAP[lang]
            lang_list.append(code + TOKEN_OFFSET + TOTAL_TOKENS)

    tokens = lang_list + tokens

    if add_bos:
        tokens.insert(0, eos_token_id)

    return tokens, lang_list

class Byt5LangTokenizer(ByT5Tokenizer):
    def __init__(self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        model_max_length=None,  # Define the maximum length for padding
        **kwargs,
    ):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.bos_token = eos_token
        self.offset = TOKEN_OFFSET

        self.pad_id = 0
        self.eos_id = 1
        self.unk_id = 2

        self.model_max_length = model_max_length
        self.special_token_start = TOKEN_OFFSET + TOTAL_TOKENS

        super().__init__()

    def __call__(self, texts: List[str] | str, langs: List[List[str]] | List[str] | None = None, pad_token_id: int = 0, **kwargs):
        tokenized = []
        all_langs = []
        attention_masks = []

        is_list = True
        # Convert to list of lists format
        if isinstance(texts, str):
            texts = [texts]
            is_list = False

        if langs is None:
            langs = [None] * len(texts)

        if isinstance(langs[0], str):
            langs = [langs]

        assert len(langs) == len(texts)

        for text, lang in zip(texts, langs):
            tokens, lang_list = _tokenize(text, lang)
            
            # Padding to fixed length (model_max_length)
            if self.model_max_length is not None:
                tokens = tokens[:self.model_max_length]  # Truncate if necessary
                attention_mask = [1] * len(tokens)  # 1 for actual tokens
                padding_length = self.model_max_length - len(tokens)
                tokens.extend([self.pad_id] * padding_length)  # Pad with pad_token_id
                attention_mask.extend([0] * padding_length)  # Pad attention mask with 0s
            else:
                attention_mask = [1] * len(tokens)  # No padding if no model_max_length specified

            tokenized.append(tokens)
            all_langs.append(lang_list)
            attention_masks.append(attention_mask)

        # Convert back to flat format
        if not is_list:
            tokenized = tokenized[0]
            all_langs = all_langs[0]
            attention_masks = attention_masks[0]

        return {
            "input_ids": tokenized,
            "langs": all_langs,
            "attention_mask": attention_masks,  # Return attention mask
        }

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, (np.ndarray, torch.Tensor)):
            token_ids = token_ids.tolist()

        token_ids = [t for t in token_ids if TOKEN_OFFSET <= t < self.special_token_start]
        token_ids = [t - TOKEN_OFFSET for t in token_ids]
        text = utf16_numbers_to_text(token_ids)
        return text

class EncoderImageProcessor(DonutImageProcessor):
    def __init__(self, *args, max_size=None, align_long_axis=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_size = kwargs.get("patch_size", (4, 4))
        self.max_size = max_size
        self.do_align_long_axis = align_long_axis

    @classmethod
    def numpy_resize(cls, image: np.ndarray, size, interpolation=cv2.INTER_LANCZOS4):
        max_width, max_height = size["width"], size["height"]

        resized_image = cv2.resize(image, (max_width, max_height), interpolation=interpolation)
        resized_image = resized_image.transpose(2, 0, 1)

        return resized_image

    def process_inner(self, images: List[np.ndarray]):
        assert images[0].shape[2] == 3 # RGB input images, channel dim last

        if self.do_align_long_axis:
            # Rotate if the bbox is wider than it is tall
            images = [EncoderImageProcessor.align_long_axis(image, size=self.max_size, input_data_format=ChannelDimension.LAST) for image in images]

            # Verify that the image is wider than it is tall
            for img in images:
                assert img.shape[1] >= img.shape[0]

        # This also applies the right channel dim format, to channel x height x width
        images = [EncoderImageProcessor.numpy_resize(img, self.max_size, self.resample) for img in images]
        assert images[0].shape[0] == 3 # RGB input images, channel dim first

        # Convert to float32 for rescale/normalize
        images = [img.astype(np.float32) for img in images]

        # Pads with 255 (whitespace)
        # Pad to max size to improve performance
        max_size = self.max_size
        images = [
            EncoderImageProcessor.pad_image(
                image=image,
                size=max_size,
                input_data_format=ChannelDimension.FIRST,
                pad_value=255
            )
            for image in images
        ]

        # Rescale and normalize
        for idx in range(len(images)):
            images[idx] = (images[idx].astype(np.float64) * self.rescale_factor).astype(np.float32)

        images = [
            EncoderImageProcessor.normalize(img, mean=self.image_mean, std=self.image_std, input_data_format=ChannelDimension.FIRST)
            for img in images
        ]

        return images

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_pad: bool = None,
        random_padding: bool = False,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        images = make_list_of_images(images)

        # Convert to numpy for later processing steps
        images = [np.array(img) for img in images]
        images = self.process_inner(images)

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @classmethod
    def pad_image(
        cls,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        output_height, output_width = size["height"], size["width"]
        input_height, input_width = get_image_size(image, channel_dim=input_data_format)

        delta_width = output_width - input_width
        delta_height = output_height - input_height

        assert delta_width >= 0 and delta_height >= 0

        pad_top = delta_height // 2
        pad_left = delta_width // 2

        pad_bottom = delta_height - pad_top
        pad_right = delta_width - pad_left

        padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        return pad(image, padding, data_format=data_format, input_data_format=input_data_format, constant_values=pad_value)

    @classmethod
    def align_long_axis(
        cls,
        image: np.ndarray,
        size: Dict[str, int],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        input_height, input_width = image.shape[:2]
        output_height, output_width = size["height"], size["width"]

        if (output_width < output_height and input_width > input_height) or (
            output_width > output_height and input_width < input_height
        ):
            image = np.rot90(image, 3)

        return image

    @classmethod
    def normalize(
        cls,
        image: np.ndarray,
        mean: Union[float, Iterable[float]],
        std: Union[float, Iterable[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        return normalize(
            image, mean=mean, std=std, data_format=data_format, input_data_format=input_data_format, **kwargs
        )
    
class DetectionModel(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor):
        return self.detector(image).logits

# class RecognizerModel(torch.nn.Module):
#     def __init__(self, encoder, text_encoder, decoder, query_token_count: int, decoder_start_token_id: int, eos_token_id: int, max_len: int = 25):
#         super().__init__()
#         self.encoder = encoder
#         self.text_encoder = text_encoder
#         self.decoder = decoder

#         self.query_token_count = query_token_count
#         self.decoder_start_token_id = decoder_start_token_id
#         self.eos_token_id = eos_token_id
#         self.max_len = max_len

#     def forward(self, pixel_values):
#         # 1. Vision Encoder
#         with torch.no_grad():
#             encoder_outputs = self.encoder(pixel_values=pixel_values)
#             encoder_hidden_states = encoder_outputs.last_hidden_state  # [B, N, D]

#             batch_size = encoder_hidden_states.size(0)
#             query_ids = torch.arange(self.query_token_count, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)

#             # 2. Text Encoder
#             text_encoder_outputs = self.text_encoder(
#                 input_ids=query_ids,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=torch.ones(encoder_hidden_states.shape[:2], device=pixel_values.device),
#                 return_dict=False,
#             )
#             text_hidden_states = text_encoder_outputs[0]

#             # 3. Decoder - fixed loop for tracing compatibility
#             decoder_input_ids = torch.full(
#                 (batch_size, 1),
#                 self.decoder_start_token_id,
#                 dtype=torch.long,
#                 device=pixel_values.device,
#             )

#             outputs = []
#             for _ in range(63):
#                 decoder_outputs = self.decoder(
#                     input_ids=decoder_input_ids,
#                     encoder_hidden_states=text_hidden_states,
#                     encoder_attention_mask=torch.ones(text_hidden_states.shape[:2], device=pixel_values.device),
#                     return_dict=False,
#                 )
#                 logits = decoder_outputs[0]
#                 next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

#                 decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
#                 outputs.append(next_token)

#         return decoder_input_ids[:, 1:]

def prepare_input(batch_langs, batch_pixel_values):
    batch_decoder_input = [[1] + lang for lang in batch_langs]
    max_input_length = max(len(tokens) for tokens in batch_decoder_input)

    # Pad decoder input to max length if needed, to ensure we can convert to a tensor
    for idx, tokens in enumerate(batch_decoder_input):
        if len(tokens) < max_input_length:
            padding_length = max_input_length - len(tokens)
            batch_decoder_input[idx] = [0] * padding_length + tokens

    batch_pixel_values = torch.tensor(np.stack(batch_pixel_values, axis=0), dtype=torch.float32)
    batch_decoder_input = torch.tensor(np.stack(batch_decoder_input, axis=0), dtype=torch.long)

    # Moving this after the padding fixes XLA recompilation issues
    batch_pixel_values = batch_pixel_values.to(batch_pixel_values.device)
    batch_decoder_input = batch_decoder_input.to(batch_pixel_values.device)

    return batch_pixel_values, batch_decoder_input

class RecognizerEncoder(torch.nn.Module):
    def __init__(self, encoder, text_encoder, query_token_count):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.query_token_count = query_token_count

    def forward(self, pixel_values):
        encoder_outputs = self.encoder(pixel_values)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        query_ids = torch.arange(self.query_token_count, device=pixel_values.device).unsqueeze(0).expand(encoder_hidden_states.size(0), -1)

        text_encoder_outputs = self.text_encoder(
            query_ids,
            None,
            None,
            encoder_hidden_states,
            None,
            False
        )
        text_hidden_states = text_encoder_outputs.hidden_states
        return text_hidden_states
    
class RecognizerDecoder(torch.nn.Module):
    def __init__(self, decoder, decoder_start_token_id, eos_token_id, pad_token_id):
        super().__init__()
        self.decoder = decoder
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def forward(self, decoder_input_ids, encoder_text_hidden_states):
        logits = decoder(
            decoder_input_ids,
            None,
            None,
            encoder_text_hidden_states,
            torch.ones(encoder_text_hidden_states.shape[:2], device=device),
            False,
            False
        ).logits
        return logits
    
if __name__ == '__main__':
    recognizer = RecognitionPredictor()
    detector = DetectionPredictor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(f">>>>>>> DEVICE: {device}")

    recognizer_loaded = RecognitionModelLoader("recognition_model")

    # image_processor = recognizer_loaded.processor().image_processor
    # image_processor.save_pretrained("models/image_processor")

    image_processor = EncoderImageProcessor.from_pretrained("models/image_processor")
    tokenizer = Byt5LangTokenizer.from_pretrained("models/tokenizer")
    
    # print(recognizer.processor.image_processor)

    # # Detection Model
    # image = Image.open("test_data/father_name_dummy.png").convert("RGB")
    # image = image.resize((1200, 1200))
    # model = DetectionModel(detector=detector.model)
    # model = model.to(device)
    # model.eval()
    # processed_image = torch.tensor(detector.processor(np.array(image)).pixel_values).to(device)
    # logits = model(processed_image)
    # traced_model = torch.jit.trace(model, processed_image)
    # traced_model.save("models/detector.pt")

    # correct_shape = [1200, 1200]
    # current_shape = list(logits.shape[2:])
    # if current_shape != correct_shape:
    #     logits = torch.nn.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

    # logits = logits.to(torch.float32).cpu().numpy()
    # traced_model = torch.jit.trace(model)

    #inputs
    image = Image.open("test_data/father_name_dummy.png").convert("RGB")

    # Recognizer Model
    encoder = recognizer.model.encoder
    text_encoder = recognizer.model.text_encoder
    decoder = recognizer.model.decoder
    query_token_count = recognizer.model.text_encoder.config.query_token_count
    decoder_start_token_id = recognizer.model.config.decoder_start_token_id
    eos_token_id = recognizer.model.config.eos_token_id
    pad_token_id = tokenizer.pad_id

    #encoder
    # encoder_model = RecognizerEncoder(
    #     encoder,
    #     text_encoder,
    #     query_token_count
    # )
    encoder_model = torch.jit.load("models/rec_encoder.pt", map_location = device)

    pixel_values = torch.tensor(image_processor(image).pixel_values)
    batch_pixel_values, batch_decoder_input_ids = prepare_input([[65555, 65546], [65555, 65546]], [pixel_values.cpu().squeeze(0), pixel_values.cpu().squeeze(0)])

    batch_pixel_values = batch_pixel_values.to(device)
    batch_size = batch_pixel_values.shape[0]

    encoder_text_hidden_states = encoder_model(batch_pixel_values)

    # # jit encoder
    # encoder_model.eval()
    # dummy_batch_pixel_values = batch_pixel_values[0].unsqueeze(0).to(device)
    # print(f">>>>>>>>>>>>>>>>>>>>>{dummy_batch_pixel_values.shape}")
    # traced_encoder = torch.jit.trace(encoder_model, dummy_batch_pixel_values)
    # traced_encoder.save("models/rec_encoder.pt")

    #decoder
    # decoder_model = RecognizerDecoder(decoder, decoder_start_token_id, eos_token_id, pad_token_id)
    decoder_model = torch.jit.load("models/rec_decoder.pt", map_location = device)
    decoder_input_ids = torch.full(
        (batch_size, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
        )
    outputs = []
    # # jit decoder
    # dummy_decoder_input_ids = decoder_input_ids[0].unsqueeze(0)
    # dummy_encoder_text_hidden_states = encoder_text_hidden_states[0].unsqueeze(0)
    # traced_decoder_model = torch.jit.trace(decoder_model, (dummy_decoder_input_ids, dummy_encoder_text_hidden_states))
    # traced_decoder_model.save("models/rec_decoder.pt")
    for _ in range(63):
        logits = decoder_model(
            decoder_input_ids,
            encoder_text_hidden_states
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print(next_token)
        if (next_token == eos_token_id).all() or (next_token == pad_token_id).all():
            break
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        outputs.append(next_token)
    print(decoder_input_ids)
    detected_text = tokenizer.batch_decode(decoder_input_ids[:, 1:].cpu())
    print(detected_text)
    































    # model = RecognizerModel(
    #     encoder=encoder,
    #     decoder=decoder,
    #     text_encoder=text_encoder,
    #     query_token_count=query_token_count,
    #     decoder_start_token_id=decoder_start_token_id,
    #     eos_token_id=eos_token_id
    # )

    # # model = torch.jit.load("models/recognizer.pt")

    # image = Image.open("test_data/father_name_dummy.png").convert('RGB')
    # pixel_values = image_processor(image).pixel_values
    # model = model.to(device)
    # pixel_values = torch.tensor(pixel_values, dtype=torch.float32, device=device)  # Ensure correct dtype and device
    # output = model(pixel_values)
    # tokenizer = Byt5LangTokenizer.from_pretrained("models/tokenizer")
    # # recognizer.processor.tokenizer.save_pretrained("models/tokenizer")
    # detected_text = tokenizer.batch_decode(output.cpu())
    # print(detected_text)
    # # # dummy_input = pixel_values[0].unsqueeze(0)
    # # # traced_model = torch.jit.trace(model, dummy_input)
    # # # traced_model.save("models/traced_model.pt")
    # # # exported_model: torch.export.ExportedProgram = torch.export.export(model, (dummy_input,))
    # # # print("model exported...")
    # # # torch.export.save(exported_model, "models/recognizer_exported.pt")

