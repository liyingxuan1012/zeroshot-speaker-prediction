import logging
from typing import List
from datetime import datetime
import os
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import load_prompt
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

logger = logging.getLogger(__name__)


class ContextExtractor:
    """Extract contexts information about the stories and characters of comics from all dialogs."""

    def __init__(self, model_name="gpt-4-0613"):
        self.chat = ChatOpenAI(
            model_name=model_name,
            model_kwargs={},
            temperature=1,
            verbose=True,
        )
        prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self._sys_prompt = SystemMessagePromptTemplate(
            prompt=load_prompt(os.path.join(prompt_dir, "context_extraction_system.yaml"))
        )
        self._human_prompt = HumanMessagePromptTemplate(
            prompt=load_prompt(os.path.join(prompt_dir, "context_extraction_human.yaml"))
        )
        self.text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=6000,
            chunk_overlap=0,
            encoding_name="cl100k_base",
            model_name="gpt-4-0314",
            separator="\n",
        )

    def __call__(self, text: str, character_names: List[str]):
        context_prompt = ChatPromptTemplate.from_messages([self._sys_prompt, self._human_prompt])
        texts = self._text_to_chunks(text)
        summarize = False
        if summarize:
            chain = load_summarize_chain(
                self.chat,
                chain_type="refine",
                question_prompt=context_prompt,
            )
            docs = self._texts_to_docs(texts)
            context = chain(
                {
                    "input_documents": docs,
                    "characters": "\n".join(character_names),
                },
                return_only_outputs=True,
            )["output_text"]
        else:
            chain = LLMChain(llm=self.chat, prompt=context_prompt, verbose=True)
            context = chain.run(
                {
                    "characters": "\n".join(character_names),
                    "text": texts[0],
                }
            )
        return context

    def _texts_to_docs(self, texts: List[str]):
        return [Document(page_content=t) for t in texts]

    def _text_to_chunks(self, text: str):
        return self.text_splitter.split_text(text)
