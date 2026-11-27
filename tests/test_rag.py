import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app.rag as rag


class RagIndexLoadingTest(unittest.TestCase):
    def test_persisted_index_is_loaded_once_and_reused(self):
        index = object()

        with TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir)
            (storage_dir / "docstore.json").touch()

            with (
                patch.object(rag, "_INDEX", None),
                patch.object(rag, "STORAGE_DIR", storage_dir),
                patch.object(rag, "Settings", SimpleNamespace(embed_model=None)),
                patch.object(rag, "get_embedding_model", return_value=object()),
                patch.object(rag, "StorageContext") as storage_context,
                patch.object(
                    rag,
                    "load_index_from_storage",
                    return_value=index
                ) as load_index,
            ):
                first = rag.initialize_index()
                second = rag.initialize_index()

        self.assertIs(first, index)
        self.assertIs(second, index)
        load_index.assert_called_once_with(storage_context.from_defaults.return_value)

    def test_whatsapp_formatting_is_only_added_to_whatsapp_prompt(self):
        index = Mock()

        with (
            patch.object(rag, "initialize_index", return_value=index),
            patch.object(rag, "get_llm"),
            patch.object(rag, "get_memory"),
        ):
            rag.create_chat_engine(
                "conversation-id",
                "model",
                whatsapp_format=True
            )
            whatsapp_prompt = index.as_chat_engine.call_args.kwargs["system_prompt"]

            rag.create_chat_engine("conversation-id", "model")
            web_prompt = index.as_chat_engine.call_args.kwargs["system_prompt"]

        self.assertIn("Do not use Markdown formatting", whatsapp_prompt)
        self.assertIn("convert each row into a clear list item", whatsapp_prompt)
        self.assertIn(
            "Do not assume retrieved legal documents belong to",
            whatsapp_prompt
        )
        self.assertIn(
            "The information I have does not contain",
            whatsapp_prompt
        )
        self.assertIn(
            'Do not say "the excerpts you provided"',
            whatsapp_prompt
        )
        self.assertNotIn("Do not use Markdown formatting", web_prompt)


if __name__ == "__main__":
    unittest.main()
