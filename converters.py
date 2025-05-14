from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.recognition.loader import RecognitionModelLoader
import torch
import numpy as np
from PIL import Image
import base64
import io

from typing import Dict, Union, Optional, List, Iterable, Any

import cv2
from torch import TensorType
from transformers import DonutImageProcessor
from transformers.image_transforms import pad, normalize
from transformers.image_utils import PILImageResampling, ImageInput, ChannelDimension, make_list_of_images, get_image_size
import numpy as np
from PIL import Image

from transformers import ByT5Tokenizer

import PIL

import warnings

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    infer_channel_dimension_format,
)
from transformers.utils import TensorType as TFTensorType


import PIL.Image
import torch

# Detection image processor
class SegformerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Segformer image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `(size["height"],
            size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
        size (`Dict[str, int]` *optional*, defaults to `{"height": 512, "width": 512}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BILINEAR`):
            Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the
            `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale`
            parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is
            used for background, and background itself is not included in all classes of a dataset (e.g. ADE20k). The
            background label will be replaced by 255. Can be overridden by the `do_reduce_labels` parameter in the
            `preprocess` method.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        **kwargs,
    ) -> None:
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` parameter is deprecated and will be removed in a future version. Please use "
                "`do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        super().__init__(**kwargs)
        size = size if size is not None else {"height": 512, "width": 512}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_reduce_labels = do_reduce_labels
        self._valid_processor_keys = [
            "images",
            "segmentation_maps",
            "do_resize",
            "size",
            "resample",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "image_mean",
            "image_std",
            "do_reduce_labels",
            "return_tensors",
            "data_format",
            "input_data_format",
        ]

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        """
        Overrides the `from_dict` method from the base class to make sure `do_reduce_labels` is updated if image
        processor is created using from_dict and kwargs e.g. `SegformerImageProcessor.from_pretrained(checkpoint,
        reduce_labels=True)`
        """
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in kwargs:
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    def _preprocess(
        self,
        image: ImageInput,
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        rescale_factor: Optional[float] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):

        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)

        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)

        return image

    def _preprocess_image(
        self,
        image: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> np.ndarray:
        """Preprocesses a single image."""
        # All transformations expect numpy arrays.
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)

        image = self._preprocess(
            image=image,
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def __call__(self, images, segmentation_maps=None, **kwargs):
        """
        Preprocesses a batch of images and optionally segmentation maps.

        Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be
        passed in as positional arguments.
        """
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: Optional[ImageInput] = None,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: Optional[bool] = None,
        return_tensors: Optional[Union[str, TFTensorType]] = None,
        data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation map to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after `resize` is applied.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation.
            do_reduce_labels (`bool`, *optional*, defaults to `self.do_reduce_labels`):
                Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0
                is used for background, and background itself is not included in all classes of a dataset (e.g.
                ADE20k). The background label will be replaced by 255.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                    - `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                    - `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)
        images = [
            self._preprocess_image(
                image=img,
                do_resize=do_resize,
                resample=resample,
                size=size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            for img in images
        ]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
    
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

class RecognizerModel(torch.nn.Module):
    def __init__(self, encoder, text_encoder, decoder, query_token_count: int, decoder_start_token_id: int, eos_token_id: int):
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder

        self.query_token_count = query_token_count
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = eos_token_id

    def forward(self, pixel_values, decoder_input_ids):
        sequence_scores = torch.zeros(pixel_values.shape[0], dtype=torch.bool, device=pixel_values.device).unsqueeze(1)
        all_done = torch.zeros(pixel_values.shape[0], dtype=torch.bool, device=pixel_values.device)
        batch_predictions = torch.zeros(pixel_values.shape[0], dtype=torch.int64, device=pixel_values.device).unsqueeze(1)
        device_pad_token = torch.tensor(0, device=pixel_values.device)
        decoder_position_ids = torch.ones_like(decoder_input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        # 1. Vision Encoder
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

        # 3. Decoder Loop (Fixed Iterations)
        for _ in range(63):
            return_dict = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=text_hidden_states,
                cache_position=decoder_position_ids
            )
            decoder_position_ids = decoder_position_ids[-1:] + 1
            logits = return_dict["logits"]
            preds = torch.argmax(logits[:, -1], dim=-1)
            scores = torch.max(torch.nn.functional.softmax(logits[:, -1], dim=-1), dim=-1).values.unsqueeze(1)
            done = (preds == 1) | (preds == 0)
            all_done = all_done | done

            scores = scores.masked_fill(all_done, 0)
            sequence_scores = torch.cat([sequence_scores, scores], dim=1)

            decoder_input_ids = preds.unsqueeze(1)
            decoder_input_ids = torch.where(all_done.unsqueeze(1), device_pad_token, decoder_input_ids)
            batch_predictions = torch.cat([batch_predictions, decoder_input_ids], dim=1)

        sequence_scores = torch.sum(sequence_scores, dim=-1) / torch.sum(sequence_scores != 0, dim=-1)
        sequence_scores = sequence_scores[:batch_size]

        return (batch_predictions[:, 1:], sequence_scores)
    
class DetectionModel(torch.nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image: torch.Tensor):
        return self.detector(image).logits

def prepare_input(batch_langs, batch_pixel_values, batch_size):
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
    
if __name__ == '__main__':
    recognizer = RecognitionPredictor()
    detector = DetectionPredictor()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # recognizer_loaded = RecognitionModelLoader("recognition_model")
    # image_processor = recognizer_loaded.processor().image_processor
    # image_processor.save_pretrained("models/image_processor")
    image_processor = EncoderImageProcessor.from_pretrained("models/image_processor")
    # print(image_processor)

    # print(recognizer.processor.image_processor)

    # # Detection Model
    # image = Image.open("test_data/499_back.png").convert("RGB")
    # image = image.resize((1200, 1200))
    # model = torch.jit.load("models/detector.pt", map_location = device)
    # model = model.to(device)
    # model.eval()
    # processor = SegformerImageProcessor.from_pretrained("models/detector_processor")
    # print(processor)
    # processor.save_pretrained("models/detector_processor")
    # processed_image = torch.tensor(detector.processor(np.array(image)).pixel_values).to(device)
    # processed_image = processor(np.array(image)).pixel_values
    # print(torch.tensor(processed_image).shape)
    # logits = detector([image])
    # print(logits)
    # traced_model = torch.jit.trace(model, processed_image)
    # traced_model.save("models/detector.pt")

    # correct_shape = [1200, 1200]
    # current_shape = list(logits.shape[2:])
    # if current_shape != correct_shape:
    #     logits = torch.nn.interpolate(logits, size=correct_shape, mode='bilinear', align_corners=False)

    # logits = logits.to(torch.float32).cpu().numpy()
    # traced_model = torch.jit.trace(model)



    # Recognizer Model
    encoder = recognizer.model.encoder
    text_encoder = recognizer.model.text_encoder
    decoder = recognizer.model.decoder
    query_token_count = recognizer.model.text_encoder.config.query_token_count
    decoder_start_token_id = recognizer.model.config.decoder_start_token_id
    eos_token_id = recognizer.model.config.eos_token_id

    model = RecognizerModel(
        encoder=encoder,
        decoder=decoder,
        text_encoder=text_encoder,
        query_token_count=query_token_count,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id
    )

    # # model = torch.jit.load("models/recognizer.pt")

    image = Image.open("test_data/father_name_dummy.png").convert('RGB')
    pixel_values = image_processor(image).pixel_values

    # output_old = recognizer([image], [['en', 'bn']], det_predictor=detector)


    # new_pixels = torch.stack([torch.tensor(pixel_values, dtype=torch.float32, device=device).squeeze(0), torch.tensor(pixel_values, dtype=torch.float32, device=device).squeeze(0)])
    # print(new_pixels.shape)

    print("[][]][][][][][][[][][][][][][][][][][][]")

    model = model.to(device)
    pixel_values = torch.tensor(pixel_values, dtype=torch.float32, device=device)  # Ensure correct dtype and device
    batch_pixels, batch_decoder_input_ids = recognizer.prepare_input([[65555, 65546]], [pixel_values.cpu().squeeze(0)], 1)
    batch_pixels = batch_pixels.to(device)
    batch_decoder_input_ids = batch_decoder_input_ids.to(device)
    print("@@@@@@@")
    print(batch_pixels.shape, batch_decoder_input_ids.shape, batch_decoder_input_ids.dtype)
    # output = model(batch_pixels, batch_decoder_input_ids)
    # print(output)
    tokenizer = Byt5LangTokenizer.from_pretrained("models/tokenizer")
    # # recognizer.processor.tokenizer.save_pretrained("models/tokenizer")
    # detected_text = tokenizer.batch_decode(output[0].cpu())
    # print(detected_text)

    model = RecognizerModel(
        encoder=encoder,
        decoder=decoder,
        text_encoder=text_encoder,
        query_token_count=query_token_count,
        decoder_start_token_id=decoder_start_token_id,
        eos_token_id=eos_token_id
    )
    model = model.to(device)
    model.eval()
    print(f"========{batch_decoder_input_ids.dtype}")
    dummy_pixels = torch.rand(1, 3, 256, 896, dtype=torch.float32, device=device)
    dummy_decoder_input_ids = torch.randint(low=0, high=100, size=(1, 63), dtype=torch.long, device=device)
    print(f"====================={dummy_pixels.shape, dummy_decoder_input_ids.shape, dummy_decoder_input_ids.dtype}")
    traced_model = torch.jit.trace(model, (dummy_pixels, dummy_decoder_input_ids))
    traced_model.save("models/recognizer.pt")

