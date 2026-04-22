from config import settings
from graph_index import build_property_graph_index
from ingestion import load_documents, split_change_orders


def main():
    documents = load_documents(settings.data_path)
    split_documents = split_change_orders(documents)
    return build_property_graph_index(split_documents)


if __name__ == "__main__":
    main()
