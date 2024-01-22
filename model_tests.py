import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))
from ragga import (  # noqa: E402
    Config,
    Encoder,
    Generator,
    MarkdownDataset,
    VectorDatabase,
    WebSearchRetriever,
)
from ragga.crafting.prompt import Llama2ChatPrompt, Phi2ChatPrompt, Phi2QAPrompt, TinyLlamaChatPrompt  # noqa: E402


def output_model_response_stream(model_response: str) -> None:
    sys.stdout.write(model_response)
    sys.stdout.flush()

models: dict[str, dict] = {
    "phi-2": {
        "path": "./models/phi-2.Q8_0.gguf",
        "quant_bits": 8,
        "context_length": 2048,
        "prompt": Phi2QAPrompt,
    },
    "phi-2-chat": {
        "path": "./models/phi-2.Q8_0.gguf",
        "quant_bits": 8,
        "context_length": 2048,
        "prompt": Phi2ChatPrompt,
    },
    "llama2": {
        "path": "./models/llama-2-7b.Q5_K_M.gguf",
        "quant_bits": 5,
        "context_length": 4096,
        "prompt": Llama2ChatPrompt,
    },
    "tinyllama": {
        "path": "./models/tinyllama-1.1b-intermediate-step-1431k-3t.Q8_0.gguf",
        "quant_bits": 8,
        "context_length": 2048,
        "prompt": TinyLlamaChatPrompt,
    },
}

datasets: dict[str, dict] = {
    "CLE_Course": {
        "path": "../ObsidianVaults/CLE_Course",
        "source": "https://github.com/MVS-99/CLE_Course",
    },
    "Cybersecurity-Notes": {
        "path": "../ObsidianVaults/Cybersecurity-Notes",
        "source": "https://github.com/Twigonometry/Cybersecurity-Notes",
    },
    "FDA-Notes": {
        "path": "../ObsidianVaults/FDA-Notes",
        "source": "https://github.com/Vuenc/FDA-Notes",
    },
    "MathWiki": {
        "path": "../ObsidianVaults/MathWiki",
        "source": "https://github.com/zhaoshenzhai/MathWiki",
    },
}

eval_qa: dict[str, list[tuple[str, str]]] = {
    "general": [
        ("What is 1 + 1?", "2"),
        ("What is the capital of France?", "Paris"),
    ],
    "CLE_Course": [
        (
            "What are the operators?",
            "Multiplication, Division, Addition, Subtraction, Unary Minus, "
            "Remainder, Prefix and Postfix, Shortcut, Logical, Bitwise, Bit Shifting, Size"
        ),
        (
            "Which bitwise operators are there?",
            "& (ampersand) -> Bitwise conjunction (AND), "
            "| (bar) -> Bitwise disjunction (OR), "
            "~ (tilde) -> Bitwise negation (NOT), "
            "^ (caret) -> Bitwise exclusive disjunction (XOR)"
        ),

    ],
    "Cybersecurity-Notes": [
        (
            "What do most assessments include?",
            "enumeration, "
            "identifying running services & the purpose of the target, "
            "re-enumerating services, "
            "finding software and hardware versions, "
            "searching for exploits in found versions, "
            "collecting credentials, "
            "exploitation, "
            "enumeration (again), "
            "persistence, "
            "privilege escalation, "
            "enumeration (again), "
            "persistence (again)"
        ),
        (
            "What is CVE-2020-12271?",
            "Pre-Authentication SQL Injection, "
            "can allow exfiltration of XG firewall-resident data, which can contain local user credentials"
        ),
    ],
    "FDA-Notes": [
        (
            "How can SVD be applied for Document Ranking?",
            "Represent documents as vectors that count word occurrences (say, the 25000 most important words "
            "in the English language). A collection of documents is then represented as a matrix. How to measure "
            "the intrinsic relevance of a document to the collection"
        ),
    ],
    "MathWiki": [
        (
            "What is the definition of an Exact Sequence?",
            r"Let $G_1,\dots,G_n$ be groups. A sequence $G_1\to G_2\to\cdots\to G_n$ of homomorphisms "
            r"$\phi_i:G_i\to G_{i+1}$ is said to beexact if $\im\phi_i=\ker\phi_{i+1}$ for all $i$. "
            r"A short exact sequence is an exact sequence $1\to A\overset{\phi}{\to}B\overset{\psi}{\to}C\to1$; that "
            r"is, a sequence for which $\phi$ is injective, $\psi$ is surjective, and $\im\phi=\ker\psi$. "
            "A long exact sequence is an exact sequence that is not short."
        ),
    ],
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    conf = Config(config_path=Path(__file__).parent / "_config.yaml")

    encoder = Encoder(conf)
    faiss_db = VectorDatabase(conf, encoder)
    # prompt = TinyLlamaChatPrompt(conf)
    # if not faiss_db.loaded_from_disk:
    #     dataset = MarkdownDataset(conf)
    #     faiss_db.documents = dataset.documents
    # # Websearch is optional
    # generator = Generator(conf, prompt, faiss_db, websearch=WebSearchRetriever)
    # generator.subscribe(output_model_response_stream)
    # QUERY = "What have I written about sonification?"
    # logging.info(f"Query: {QUERY}")
    # response = generator.get_answer(QUERY)
    # logging.debug(response)
    # # /again
    # # /more or /continue
    # while True:
    #     try:
    #         query = input("Enter query: ")
    #         logging.info(f"Query: {query}")
    #         generator.get_answer(query)
    #     except KeyboardInterrupt:
    #         logging.info("Exiting...")
    #         break




