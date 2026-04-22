import re
from typing import List

from llama_index.core import Document, SimpleDirectoryReader


CHANGE_ORDER_SEPARATOR = r"!@#\$%\^&\*"


def load_documents(data_path: str) -> List[Document]:
    return SimpleDirectoryReader(data_path).load_data()


def split_change_orders(documents: List[Document]) -> List[Document]:
    split_docs: List[Document] = []
    for doc in documents:
        raw_segments = re.split(CHANGE_ORDER_SEPARATOR, doc.text.strip())
        for index, segment in enumerate(raw_segments):
            segment = segment.strip()
            if not segment:
                continue
            metadata = {
                "source_file": doc.metadata.get("file_name", "unknown"),
                "change_order_index": index + 1,
                **doc.metadata,
            }
            split_docs.append(
                Document(
                    text=segment,
                    metadata=metadata,
                    id_=f"{metadata['source_file']}_CO_{index + 1}",
                )
            )
    return split_docs
