import os
from dotenv import load_dotenv
from unstructured_ingest.logger import logger
from typing import TYPE_CHECKING, Any, Literal, Optional

from unstructured_ingest.v2.pipeline.pipeline import Pipeline
from unstructured_ingest.v2.interfaces import ProcessorConfig
from unstructured_ingest.v2.processes.connectors.local import (
    LocalIndexerConfig,
    LocalDownloaderConfig,
    LocalConnectionConfig,
    LocalUploaderConfig,
)
from unstructured_ingest.embed.interfaces import (
    BaseEmbeddingEncoder,
)
from unstructured_ingest.v2.processes.partitioner import PartitionerConfig
from unstructured_ingest.v2.processes.chunker import ChunkerConfig
from unstructured_ingest.v2.processes.embedder import EmbedderConfig

from pydantic import Field, SecretStr

from unstructured.embed.openai import OpenAIEmbeddingConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from unstructured_ingest.v2.errors import (
    ProviderError,
    QuotaError,
    RateLimitError,
    UserAuthError,
    UserError,
)

load_dotenv()


class TempOpenAIEmbeddingConfig(OpenAIEmbeddingConfig):
    api_key: SecretStr
    embedder_model_name: str = Field(
        default="text-embedding-ada-002", alias="model_name"
    )
    batch_size: Optional[int] = Field(
        default=100, description="Optional batch size for embedding requests."
    )

    def wrap_error(self, e: Exception) -> Exception:
        # https://platform.openai.com/docs/guides/error-codes/api-errors
        from openai import APIStatusError

        if not isinstance(e, APIStatusError):
            logger.error(f"unhandled exception from openai: {e}", exc_info=True)
            raise e
        error_code = e.code
        if 400 <= e.status_code < 500:
            # user error
            if e.status_code == 401:
                return UserAuthError(e.message)
            if e.status_code == 429:
                # 429 indicates rate limit exceeded and quote exceeded
                if error_code == "insufficient_quota":
                    return QuotaError(e.message)
                else:
                    return RateLimitError(e.message)
            return UserError(e.message)
        if e.status_code >= 500:
            return ProviderError(e.message)
        logger.error(f"unhandled exception from openai: {e}", exc_info=True)
        return e

    def get_client(self):
        """Creates a langchain OpenAI python client to embed elements."""
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ["EMBEDDINGS_API_KEY"],
            base_url="http://10.69.1.169:5002/v1",
        )

    def get_async_client(self) -> "AsyncOpenAI":
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=os.environ["EMBEDDINGS_API_KEY"],
            base_url="http://10.69.1.169:5002/v1",
        )


class TempEmbedderConfig(EmbedderConfig):
    embedding_provider: Optional[
        Literal[
            "openai",
            "azure-openai",
            "huggingface",
            "bedrock",
            "vertexai",
            "voyageai",
            "octoai",
            "mixedbread-ai",
            "togetherai",
        ]
    ] = Field(default=None, description="Type of the embedding class to be used.")
    embedding_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the embedding model, for the case an API key is needed.",
    )
    embedding_model_name: Optional[str] = Field(
        default=None,
        description="Embedding model name, if needed. "
        "Chooses a particular LLM between different options, to embed with it.",
    )
    embedding_aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key used for AWS-based embedders, such as bedrock",
    )
    embedding_aws_secret_access_key: Optional[SecretStr] = Field(
        default=None,
        description="AWS secret key used for AWS-based embedders, such as bedrock",
    )
    embedding_aws_region: Optional[str] = Field(
        default="us-west-2",
        description="AWS region used for AWS-based embedders, such as bedrock",
    )
    embedding_azure_endpoint: Optional[str] = Field(
        default=None,
        description="Your Azure endpoint, including the resource, "
        "e.g. `https://example-resource.azure.openai.com/`",
    )
    embedding_azure_api_version: Optional[str] = Field(
        description="Azure API version", default=None
    )

    def get_openai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.openai import OpenAIEmbeddingEncoder
        return OpenAIEmbeddingEncoder(
            config=TempOpenAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_azure_openai_embedder(
        self, embedding_kwargs: dict
    ) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.azure_openai import (
            AzureOpenAIEmbeddingConfig,
            AzureOpenAIEmbeddingEncoder,
        )

        config_kwargs = {
            "api_key": self.embedding_api_key,
            "azure_endpoint": self.embedding_azure_endpoint,
        }
        if api_version := self.embedding_azure_api_version:
            config_kwargs["api_version"] = api_version
        if model_name := self.embedding_model_name:
            config_kwargs["model_name"] = model_name

        return AzureOpenAIEmbeddingEncoder(
            config=AzureOpenAIEmbeddingConfig.model_validate(config_kwargs)
        )

    def get_octoai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.octoai import (
            OctoAiEmbeddingConfig,
            OctoAIEmbeddingEncoder,
        )

        return OctoAIEmbeddingEncoder(
            config=OctoAiEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_bedrock_embedder(self) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.bedrock import (
            BedrockEmbeddingConfig,
            BedrockEmbeddingEncoder,
        )

        return BedrockEmbeddingEncoder(
            config=BedrockEmbeddingConfig(
                aws_access_key_id=self.embedding_aws_access_key_id,
                aws_secret_access_key=self.embedding_aws_secret_access_key.get_secret_value(),
                region_name=self.embedding_aws_region,
            )
        )

    def get_vertexai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.vertexai import (
            VertexAIEmbeddingConfig,
            VertexAIEmbeddingEncoder,
        )

        return VertexAIEmbeddingEncoder(
            config=VertexAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_voyageai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.voyageai import (
            VoyageAIEmbeddingConfig,
            VoyageAIEmbeddingEncoder,
        )

        return VoyageAIEmbeddingEncoder(
            config=VoyageAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_mixedbread_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.mixedbreadai import (
            MixedbreadAIEmbeddingConfig,
            MixedbreadAIEmbeddingEncoder,
        )

        return MixedbreadAIEmbeddingEncoder(
            config=MixedbreadAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_togetherai_embedder(self, embedding_kwargs: dict) -> "BaseEmbeddingEncoder":
        from unstructured_ingest.embed.togetherai import (
            TogetherAIEmbeddingConfig,
            TogetherAIEmbeddingEncoder,
        )

        return TogetherAIEmbeddingEncoder(
            config=TogetherAIEmbeddingConfig.model_validate(embedding_kwargs)
        )

    def get_embedder(self) -> "BaseEmbeddingEncoder":
        kwargs: dict[str, Any] = {}
        if self.embedding_api_key:
            kwargs["api_key"] = self.embedding_api_key.get_secret_value()
        if self.embedding_model_name:
            kwargs["model_name"] = self.embedding_model_name
        # TODO make this more dynamic to map to encoder configs
        if self.embedding_provider == "openai":
            return self.get_openai_embedder(embedding_kwargs=kwargs)

        if self.embedding_provider == "huggingface":
            return self.get_huggingface_embedder(embedding_kwargs=kwargs)

        if self.embedding_provider == "octoai":
            return self.get_octoai_embedder(embedding_kwargs=kwargs)

        if self.embedding_provider == "bedrock":
            return self.get_bedrock_embedder()

        if self.embedding_provider == "vertexai":
            return self.get_vertexai_embedder(embedding_kwargs=kwargs)

        if self.embedding_provider == "voyageai":
            return self.get_voyageai_embedder(embedding_kwargs=kwargs)
        if self.embedding_provider == "mixedbread-ai":
            return self.get_mixedbread_embedder(embedding_kwargs=kwargs)
        if self.embedding_provider == "togetherai":
            return self.get_togetherai_embedder(embedding_kwargs=kwargs)
        if self.embedding_provider == "azure-openai":
            return self.get_azure_openai_embedder(embedding_kwargs=kwargs)

        raise ValueError(f"{self.embedding_provider} not a recognized encoder")


# Chunking and embedding are optional.
# INPUT_DIR = os.getenv("LOCAL_FILE_INPUT_DIR")
INPUT_DIR = "/app/docs/python/3.13/library"
OUTPUT_DIR = "/app/processed_docs/python/3.13/library"
if __name__ == "__main__":
    Pipeline.from_configs(
        context=ProcessorConfig(),
        indexer_config=LocalIndexerConfig(input_path=INPUT_DIR),
        downloader_config=LocalDownloaderConfig(),
        source_connection_config=LocalConnectionConfig(),
        partitioner_config=PartitionerConfig(
            partition_by_api=True,
            # api_key=SecretStr("1234"),
            api_key="1234",
            partition_endpoint="http://10.69.1.169:9025/general/v0/general",
            strategy="hi_res",
            additional_partition_args={
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "split_pdf_concurrency_level": 15,
            },
        ),
        # chunker_config=ChunkerConfig(chunking_strategy="by_title", chunk_by_api=True, chunk_api_key=SecretStr(os.environ["EMBEDDINGS_API_KEY"]), chunking_endpoint="http://10.69.1.169:9025/general/v0/general"),
        chunker_config=ChunkerConfig(
            chunking_strategy="by_title",
            chunk_by_api=True,
            chunk_api_key=os.environ["EMBEDDINGS_API_KEY"],
            chunking_endpoint="http://10.69.1.169:9025/general/v0/general",
        ),
        embedder_config=TempEmbedderConfig(
            embedding_provider="openai",
            embedding_api_key=SecretStr(os.environ["EMBEDDINGS_API_KEY"]),
            api_key=SecretStr(os.environ["EMBEDDINGS_API_KEY"]),
            embedding_model_name="dunzhang_stella_en_1.5B_v5",
        ),
        uploader_config=LocalUploaderConfig(output_dir=OUTPUT_DIR),
    ).run()


config = TempEmbedderConfig(
    embedding_provider="openai",
    embedding_api_key=SecretStr(os.environ["EMBEDDINGS_API_KEY"]),
    api_key=SecretStr(os.environ["EMBEDDINGS_API_KEY"]),
    embedding_model_name="dunzhang_stella_en_1.5B_v5",
)
embedder = config.get_embedder()
resp = embedder.embed_query("test")
