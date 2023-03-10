import json
from typing import Optional

import numpy as np
from monai.config import KeysCollection, PathLike
from monai.data.image_reader import ImageReader
from monai.transforms.transform import MapTransform, Randomizable, Transform
from transformers import CLIPTokenizer


class LoadJSON(Transform):
    """Transformation to load a JSON file."""

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
    """
    Transform to randomly select a number of sentences from a list of sentences and concatenate them into a single
        string.

    """

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

    def randomize(self, list_of_sentences_len: int) -> None:
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
    """Transformation to apply the CLIP tokenizer."""

    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")

    def __call__(self, text_input: str):
        tokenized_sentence = self.tokenizer(
            text_input,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_sentence.input_ids


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
