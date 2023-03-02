import json
from typing import Optional

import numpy as np
from monai.config import KeysCollection, PathLike
from monai.data.image_reader import ImageReader
from monai.transforms.transform import MapTransform, Randomizable, Transform
from transformers import CLIPTokenizer

# from typing import Callable, Hashable, Mapping, Optional, Sequence
# import torch
# from monai.transforms.transform import RandomizableTransform
# from monai.utils import ensure_tuple_rep


class LoadJSON(Transform):
    def __call__(self, filename: PathLike):
        with open(str(filename)) as json_file:
            data = json.load(json_file)
        return data


class LoadJSONd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadJSON(*args, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            data = self._loader(d[key])
            d[key] = data

        return d


class RandomSelectExcerptd(Randomizable, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        sentence_key: str,
        max_n_sentences: int,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.sentence_key = sentence_key
        self.max_n_sentences = max_n_sentences
        self._selected_sentence_list = [0]
        self._n_sentences = 0

    def randomize(self, list_of_sentences_len) -> None:
        n_possible_sentences = min(list_of_sentences_len, self.max_n_sentences)
        if n_possible_sentences > 1:
            self._n_sentences = np.random.randint(1, n_possible_sentences)
        else:
            self._n_sentences = 1
        sentences_list = np.arange(list_of_sentences_len)
        np.random.shuffle(sentences_list)
        self._selected_sentence_list = sentences_list[: self._n_sentences]

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key in self.key_iterator(d):
            list_of_sentences = d[key][self.sentence_key]
            self.randomize(len(list_of_sentences))
            excerpt_encoded = ""
            for i in self._selected_sentence_list:
                if excerpt_encoded == "":
                    excerpt_encoded = excerpt_encoded + list_of_sentences[i]
                else:
                    excerpt_encoded = excerpt_encoded + " " + list_of_sentences[i]
            d[key] = excerpt_encoded
        return d


class ApplyTokenizer(Transform):
    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")

    def __call__(self, tokenized_sentence):
        text_inputs = self.tokenizer(
            tokenized_sentence,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids


class ApplyTokenizerd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._padding = ApplyTokenizer(*args, **kwargs)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key in self.key_iterator(d):
            data = self._padding(d[key])
            d[key] = data

        return d


#
# class Lambda(Transform):
#     def __init__(self, func: Callable | None = None) -> None:
#         if func is not None and not callable(func):
#             raise TypeError(f"func must be None or callable but is {type(func).__name__}.")
#         self.func = func
#
#     def __call__(self, img, func):
#         fn = func if func is not None else self.func
#         if not callable(fn):
#             raise TypeError(f"func must be None or callable but is {type(fn).__name__}.")
#         out = fn(img)
#         return out
#
#
# class Lambdad(MapTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         func: Sequence[Callable] | Callable,
#         overwrite: Sequence[bool] | bool | Sequence[str] | str = True,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         super().__init__(keys, allow_missing_keys)
#         self.func = ensure_tuple_rep(func, len(self.keys))
#         self.overwrite = ensure_tuple_rep(overwrite, len(self.keys))
#         self._lambd = Lambda()
#
#     def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
#         d = dict(data)
#         for key, func, overwrite in self.key_iterator(d, self.func, self.overwrite):
#             ret = self._lambd(img=d[key], func=func)
#             if overwrite and isinstance(overwrite, bool):
#                 d[key] = ret
#             elif isinstance(overwrite, str):
#                 d[overwrite] = ret
#         return d
#
#
# class RandLambdad(Lambdad, RandomizableTransform):
#     def __init__(
#         self,
#         keys: KeysCollection,
#         func: Sequence[Callable] | Callable,
#         overwrite: Sequence[bool] | bool = True,
#         prob: float = 1.0,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         Lambdad.__init__(
#             self=self,
#             keys=keys,
#             func=func,
#             overwrite=overwrite,
#             allow_missing_keys=allow_missing_keys,
#         )
#         RandomizableTransform.__init__(self=self, prob=prob, do_transform=True)
#
#     def __call__(self, data):
#         self.randomize(data)
#         d = dict(data)
#         for key, func, overwrite in self.key_iterator(d, self.func, self.overwrite):
#             ret = d[key]
#             if self._do_transform:
#                 ret = self._lambd(ret, func=func)
#             if overwrite:
#                 d[key] = ret
#         return d
