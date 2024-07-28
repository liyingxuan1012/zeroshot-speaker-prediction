import os
import shutil
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts import load_prompt
from joblib import Parallel, delayed
from typing import List
import string


class LLMSpeakerPredictor:
    """Speaker prediction using LLM."""

    def __init__(self, character_names: List[str], context: str, config: dict) -> None:
        self.config = config
        self.character_names = character_names
        self.character_ids = [string.ascii_letters[i] for i in range(len(character_names))]
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=config.chunk_size,
            chunk_overlap=10,
            length_function=lambda x: x.count("\n"),
        )
        self.context = context if config.use_context else None
        self.system_prompt_path = config.prompt.system
        self.system_prompt_path_candidate = config.prompt.system_candidate
        self.print_confidence = config.print_confidence
        self.model_name = config.model
        assert os.path.exists(self.system_prompt_path)
        assert os.path.exists(self.system_prompt_path_candidate)

    def __call__(
        self,
        texts: List[str],
        character_candidates: List[List[str]] = None,
        confidences: List[float] = None,
    ):
        prompt_system = self._get_system_prompt(
            self.character_names, self.character_ids, character_candidates is not None
        )
        prompt_user = self._get_user_prompt(texts, character_candidates, confidences)

        chunked_texts = self.text_splitter.split_text(prompt_user)

        outputs = Parallel(n_jobs=4, verbose=1, backend="multiprocessing")(
            delayed(_llm_predict)(self.model_name, prompt_system, chunk, i)
            for i, chunk in enumerate(chunked_texts)
        )

        # Parse LLM output
        character_id_column_index = 2
        confidence_index = 3
        text_character_ids = [None] * len(texts)
        confidences = [0] * len(texts)
        for o in outputs[::-1]:
            for line in o.split("\n"):
                # Split line by "|"
                # Format of line: 'index | text | character_id | confidence'
                items = line.strip().split("|")
                if len(items) < 4:
                    continue

                # First column is the index of the text
                try:
                    index = int(items[0]) - 1
                except ValueError:
                    continue
                if index >= len(texts):
                    print(f"index {index} is out of range")
                    continue

                # Third column is the character id
                character_id = items[character_id_column_index].strip()
                if not character_id in self.character_ids:
                    continue
                text_character_ids[index] = self.character_ids.index(character_id)

                # Fourth column is the confidence
                try:
                    confidences[index] = int(items[confidence_index].strip())
                except ValueError:
                    pass
        return text_character_ids, confidences

    def _get_user_prompt(self, texts, character_candidates, confidences) -> str:
        lines = []
        for i, text in enumerate(texts):
            line = f"{i+1} | {text}"
            if character_candidates is not None:
                if len(character_candidates[i]) == 0:
                    line += f" | -"
                else:
                    for c in character_candidates[i]:
                        line += f" | {self.character_names[c]}"
                    if self.print_confidence:
                        line += f" ({confidences[i]:.2f})"
            lines.append(line)
        return "\n".join(lines)

    def _get_system_prompt(self, character_names, character_ids, use_character_candidates) -> str:
        template_path = (
            self.system_prompt_path_candidate
            if use_character_candidates
            else self.system_prompt_path
        )
        template = SystemMessagePromptTemplate(prompt=load_prompt(template_path))
        characters = "\n".join(
            [f"{id} | {name}" for name, id in zip(character_names, character_ids)]
        )
        if self.context is None:
            return template.format(characters=characters).content
        else:
            return template.format(characters=characters, context=self.context).content


def _llm_predict(model_name, prompt_system, chunked_prompt_user, chunk_index, num_tries=3):
    chat = ChatOpenAI(
        model_name=model_name,
        temperature=1,
        verbose=True,
        max_retries=20,
    )
    chain = LLMChain(
        llm=chat,
        prompt=ChatPromptTemplate.from_messages(
            [SystemMessage(content=prompt_system), HumanMessage(content=chunked_prompt_user)]
        ),
        verbose=True,
    )

    for i in range(num_tries):
        result = chain.run({})
        print(result)
        n_input = ["|" in l for l in chunked_prompt_user.split("\n")].count(True)
        n_output = ["|" in l for l in result.split("\n")].count(True)
        if n_input == n_output:
            break
        print(f"{i}: Number of input and output lines are different: {n_input} != {n_output}")
    return result
