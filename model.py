import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from ragga import ChatPrompt, Config, Encoder, Generator, MarkdownDataset, Search, VectorDatabase  # noqa: E402


def output_model_response_stream(model_response: str) -> None:
    sys.stdout.write(model_response)
    sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search = Search()
    conf = Config(config_path=Path(__file__).parent / "_config.yaml")
    encoder = Encoder(conf)
    faiss_db = VectorDatabase(conf, encoder)
    prompt = ChatPrompt(conf)
    if not faiss_db.loaded_from_disk:
        dataset = MarkdownDataset(conf)
        faiss_db.documents = dataset.documents
    generator = Generator(conf, prompt, faiss_db)
    generator.subscribe(output_model_response_stream)
    QUERY = "What have I written about sonification?"
    logging.info(f"Query: {QUERY}")
    generator.get_answer(QUERY)
    # /again
    # /more or /continue
    while True:
        try:
            query = input("Enter query: ")
            logging.info(f"Query: {query}")
            generator.get_answer(query)
        except KeyboardInterrupt:
            logging.info("Exiting...")
            break




