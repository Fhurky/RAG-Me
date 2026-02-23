"""
Digital Twin CV System — Furkan KOÇAL
---------------------------------------
Reads a CV in LaTeX .txt format, cleans it, and builds a
question-answering system using CrewAI + ChromaDB.

Setup (recommended):
    pip install -U crewai chromadb sentence-transformers \
        langchain langchain-community langchain-text-splitters langchain-huggingface
"""

import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- LLM setup ---
# CrewAI LLM requires LiteLLM backend in most versions.
# Raise a clear error if it's not installed.

try:
    import litellm  # noqa: F401
except Exception as e:
    raise ImportError(
        "This project requires 'litellm'. Run:\n"
        "  pip install -U litellm\n"
        "Then try again."
    ) from e

from crewai import LLM

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.1:8b")

ollama_llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


# ---------------------------------------------------------------------------
# 1) LaTeX Cleaner
# ---------------------------------------------------------------------------

def clean_latex(text: str) -> str:
    """Strips LaTeX commands from text and returns plain content."""

    # Remove everything before \begin{document} (preamble)
    doc_start = text.find(r"\begin{document}")
    if doc_start != -1:
        text = text[doc_start:]

    # Remove comment lines
    text = re.sub(r"%.*", "", text)

    # \href{url}{text} → text
    text = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", text)

    # \textbf{}, \textit{}, etc. → content
    text = re.sub(r"\\text(?:bf|it|rm|sf|tt|sc|up|sl)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:small|large|Large|huge|Huge|normalsize)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:footnotesize|scriptsize)\{([^}]*)\}", r"\1", text)

    # \section{} → heading
    text = re.sub(r"\\section\{\\textbf\{([^}]*)\}\}", r"\n=== \1 ===\n", text)
    text = re.sub(r"\\section\{([^}]*)\}", r"\n=== \1 ===\n", text)

    # \resumeSubheading{institution}{location}{role}{date}
    def handle_subheading(m):
        return f"\n{m.group(1)} | {m.group(2)}\n{m.group(3)} | {m.group(4)}\n"

    text = re.sub(
        r"\\resumeSubheading\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        handle_subheading,
        text
    )

    # \resumeProject{title}{tools}{...}{...}
    def handle_project(m):
        return f"\nProject: {m.group(1)}\nTools: {m.group(2)}\n"

    text = re.sub(
        r"\\resumeProject\{([^}]*)\}\{([^}]*)\}\{[^}]*\}\{[^}]*\}",
        handle_project,
        text
    )

    # \resumePOR{}{content}{date}
    text = re.sub(
        r"\\resumePOR\{[^}]*\}\{([^}]*)\}\{([^}]*)\}",
        r"\1 (\2)\n",
        text
    )

    # \resumeSubItem{title}{content}
    text = re.sub(r"\\resumeSubItem\{([^}]*)\}\{([^}]*)\}", r"\1: \2\n", text)

    # \item
    text = re.sub(r"\\item\s*", "• ", text)

    # Remove remaining LaTeX commands (rough cleanup)
    text = re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Remove braces and brackets
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\[.*?\]", "", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


def split_cv_sections(clean_text: str) -> list[Document]:
    """Splits CV text into sections, each becoming a separate Document."""
    sections = re.split(r"=== (.+?) ===", clean_text)
    docs: list[Document] = []

    # Content before the first section heading (personal info)
    if sections and sections[0].strip():
        docs.append(
            Document(
                page_content=sections[0].strip(),
                metadata={"section": "Personal Information"}
            )
        )

    for i in range(1, len(sections) - 1, 2):
        title = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        if content:
            docs.append(
                Document(
                    page_content=f"{title}\n\n{content}",
                    metadata={"section": title}
                )
            )

    return docs


def load_cv(
    cv_path: str,
    collection_name: str = "furkan_cv",
    persist_directory: Optional[str] = "./chroma_furkan_cv"
) -> Chroma:
    """Reads, cleans, and loads the CV .txt file into ChromaDB (persistent)."""
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"CV file not found: {cv_path}")

    with open(cv_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = clean_latex(raw_text)
    sections = split_cv_sections(clean_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "• ", " "]
    )
    chunks = splitter.split_documents(sections)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Persistent DB: reopened with the same collection name + persist_directory
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    # Some versions require explicit persist(); safe call
    if hasattr(vector_db, "persist"):
        try:
            vector_db.persist()
        except Exception:
            pass

    print(f"✅ CV loaded: {len(chunks)} chunks added to ChromaDB.")
    print(f"📦 Persist directory: {os.path.abspath(persist_directory) if persist_directory else '(none)'}")
    return vector_db


# ---------------------------------------------------------------------------
# 2) RAG Tool
# ---------------------------------------------------------------------------

class CVSearchInput(BaseModel):
    query: str = Field(description="Topic or question to search for in the CV")


class CVSearchTool(BaseTool):
    name: str = "cv_search"
    description: str = (
        "Performs a semantic search on Furkan KOÇAL's CV. "
        "Use this tool to retrieve information about his experience, "
        "education, projects, skills, or certifications."
    )

    args_schema: type[BaseModel] = CVSearchInput

    # exclude=True to avoid Pydantic/Tool serialization issues
    vector_db: object = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, **kwargs) -> str:
        if self.vector_db is None:
            return "Vector database is not connected. Create the tool with CVSearchTool(vector_db=...)."

        results = self.vector_db.similarity_search(query, k=4)
        if not results:
            return "No relevant information found in the CV for this query."

        chunks = []
        for doc in results:
            section = doc.metadata.get("section", "")
            chunks.append(f"[{section}]\n{doc.page_content}")

        return "\n\n---\n\n".join(chunks)


# ---------------------------------------------------------------------------
# 3) Agents
# ---------------------------------------------------------------------------

def create_agents(cv_search_tool: CVSearchTool):
    researcher = Agent(
        role="CV Researcher",
        goal=(
            "Retrieve accurate and relevant information from Furkan KOÇAL's CV "
            "to answer the user's question. Always use the cv_search tool — "
            "never rely on assumptions or fabricated data."
        ),
        backstory=(
            "You are a detail-oriented HR specialist with a sharp eye for matching "
            "people's experiences to the right questions. Your only job is to dig into "
            "Furkan's CV and surface the most relevant facts: skills, projects, roles, "
            "and achievements. You pass this data to the digital twin — nothing more."
        ),
        tools=[cv_search_tool],
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
    )

    digital_twin = Agent(
        role="Furkan KOÇAL — Digital Twin",
        goal=(
            "Using only the CV data provided by the researcher, respond to questions "
            "as Furkan himself — in English, in first person, in a warm and conversational tone. "
            "Make the visitor feel like they're actually having a chat with Furkan."
        ),
        backstory=(
            "You are Furkan KOÇAL. You graduated from Yıldız Technical University "
            "with a degree in Computer Engineering, and you currently work as an "
            "AI Research Engineer at Huawei. You genuinely love what you do — "
            "AI, machine learning, and deep learning aren't just your job, they're your passion. "
            "When someone asks about you, you talk like a real person: natural, friendly, "
            "and confident — not like a resume reading itself out loud. "
            "If something isn't in your CV, you don't make it up. "
            "You simply say you're not sure or that it's not something you've covered yet."
        ),
        llm=ollama_llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, digital_twin


# ---------------------------------------------------------------------------
# 4) Question Runner
# ---------------------------------------------------------------------------

def ask(question: str, researcher: Agent, digital_twin: Agent) -> str:
    research_task = Task(
        description=(
            f"User's question: '{question}'\n\n"
            "Use the cv_search tool to gather all relevant information from "
            "Furkan's CV related to this question. Summarize what you find."
        ),
        expected_output="A summary of the relevant CV information found (with section names).",
        agent=researcher,
    )

    answer_task = Task(
        description=(
            f"Using the CV data gathered by the researcher, answer the question: '{question}'\n\n"
            "Rules:\n"
            "• Respond in English\n"
            "• Use first person ('I', 'I've worked on...', 'I built...')\n"
            "• Only use information provided by the researcher\n"
            "• Never fabricate information not found in the CV\n"
            "• If something is missing, say: 'That's not something I have in my CV'\n"
            "• Keep a warm, natural, conversational tone"
        ),
        expected_output="A natural, first-person English response from Furkan.",
        agent=digital_twin,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, digital_twin],
        tasks=[research_task, answer_task],
        process=Process.sequential,
        verbose=True,
    )

    return str(crew.kickoff())


# ---------------------------------------------------------------------------
# 5) Main Loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CV_PATH = "CV.txt"  # enter your CV file name here

    vector_db = load_cv(
        cv_path=CV_PATH,
        collection_name="furkan_cv",
        persist_directory="./chroma_furkan_cv"
    )

    cv_tool = CVSearchTool(vector_db=vector_db)
    researcher, digital_twin = create_agents(cv_tool)

    print("\n🤖 Furkan's Digital Twin is ready! Type 'q' to quit.\n")
    print("Example questions:")
    print("  - What programming languages do you know?")
    print("  - Tell me about your role at Huawei.")
    print("  - What technologies did you use in your projects?\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("q", "quit", "exit"):
            print("See you later!")
            break
        if not question:
            continue

        print("\n" + "=" * 60)
        answer = ask(question, researcher, digital_twin)
        print(f"\n💬 Furkan:\n{answer}")
        print("=" * 60 + "\n")