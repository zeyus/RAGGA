import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from ragga import Config, Encoder, Generator, MarkdownDataset, Search, VectorDatabase, simple_prompt  # noqa: E402

if __name__ == "__main__":
    search = Search()
    conf = Config(config_path=Path(__file__).parent / "_config.yaml")
    encoder = Encoder(conf)
    faiss_db = VectorDatabase(conf, encoder)
    dataset = MarkdownDataset(conf)
    faiss_db.documents = dataset.documents

    generator = Generator(conf, simple_prompt, faiss_db)

    QUERY = "What have I written about my IMC work?"
    generator.get_answer(QUERY)


