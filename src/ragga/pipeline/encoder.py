from types import MappingProxyType

from langchain.embeddings import HuggingFaceEmbeddings

from ragga.core.config import Config, Configurable


class Encoder(Configurable):
    """Encoder to create word embeddings"""
    _config_key = "encoder"

    _default_config = MappingProxyType({
        "model_path": "sentence-transformers/all-MiniLM-l6-v2",
    })

    _default_model_kwargs = MappingProxyType({
        "device": "cuda",
    })

    _default_encode_kwargs = MappingProxyType({
        "normalize_embeddings": False,
    })

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)

        self._merge_default_kwargs(dict(self._default_model_kwargs), "model_kwargs")
        self._merge_default_kwargs(dict(self._default_encode_kwargs), "encode_kwargs")

        self.encoder = HuggingFaceEmbeddings(
            model_name=self.config[self._config_key]["model_path"],
            model_kwargs=self.config[self._config_key]["model_kwargs"],
            encode_kwargs=self.config[self._config_key]["encode_kwargs"],
        )
