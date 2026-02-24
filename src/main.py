"""
Digital Twin CV System — Furkan KOÇAL (TR)
-----------------------------------------
Reads a CV in LaTeX .txt format, cleans it more safely,
chunks adaptively by section, indexes into ChromaDB, and
answers questions using CrewAI + MMR retrieval with metadata filtering.

Setup (recommended):
    pip install -U crewai chromadb sentence-transformers \
        langchain langchain-community langchain-text-splitters langchain-huggingface litellm
Optional (better LaTeX text extraction):
    pip install -U pylatexenc
"""

import os
import re
import json
import hashlib
from typing import Optional, Any

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- LLM backend check (CrewAI often needs LiteLLM) ---
try:
    import litellm  # noqa: F401
except Exception as e:
    raise ImportError(
        "Bu proje 'litellm' gerektiriyor. Kur:\n"
        "  pip install -U litellm\n"
        "Sonra tekrar dene."
    ) from e

from crewai import LLM

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
# Not: "gemini/" prefix şart. Prefix yoksa LiteLLM bunu Vertex AI sanıp GCP credential ister.
gemini_llm = LLM(model=GEMINI_MODEL)


# ---------------------------------------------------------------------------
# 0) Utilities
# ---------------------------------------------------------------------------

def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_intro_question(q: str) -> bool:
    q = q.lower().strip()
    patterns = [
        r"\bben kimim\b",
        r"\bkendini tan[ıi]t\b",
        r"\bk[ıi]saca kendini\b",
        r"\bhakk[ıi]nda\b.*\b[öo]zet\b",
        r"\b[öo]zetle\b",
        r"\bprofil\b",
        r"\bbiografi\b",
        r"\bkimdir\b",
    ]
    return any(re.search(p, q) for p in patterns)


def infer_section_filter(query: str) -> Optional[str]:
    """
    Heuristic: sorunun içeriğine göre section filtresi öner.
    Bu filtre 'zorunlu' değil; MMR'a yardımcı olmak için kullanıyoruz.
    """
    q = query.lower()

    # TR/EN keyword karışımı tolere
    if any(k in q for k in ["huawei", "iş", "work", "deneyim", "experience", "pozisyon", "role"]):
        return "Experience"
    if any(k in q for k in ["eğitim", "education", "üniversite", "yıldız", "ytu", "gpa", "ortalama"]):
        return "Education"
    if any(k in q for k in ["proje", "project", "srgan", "inpainting", "emotion", "lstm", "portfolyo", "portfolio"]):
        return "Projects"
    if any(k in q for k in ["sertifika", "certificate", "coursera", "ibm", "nvidia"]):
        return "Certifications"
    if any(k in q for k in ["yetenek", "skill", "teknoloji", "stack", "python", "pytorch", "tensorflow", "sql"]):
        return "Skills"
    return None


# ---------------------------------------------------------------------------
# 1) Safer LaTeX Cleaner
# ---------------------------------------------------------------------------

_LATEX_MACRO_REWRITES = [
    # \href{url}{text} -> text
    (re.compile(r"\\href\{[^}]*\}\{([^}]*)\}"), r"\1"),
    # \textbf{...} etc -> content
    (re.compile(r"\\text(?:bf|it|rm|sf|tt|sc|up|sl)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:small|large|Large|huge|Huge|normalsize)\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:footnotesize|scriptsize)\{([^}]*)\}"), r"\1"),
    # sections -> heading markers
    (re.compile(r"\\section\{\\textbf\{([^}]*)\}\}"), r"\n=== \1 ===\n"),
    (re.compile(r"\\section\{([^}]*)\}"), r"\n=== \1 ===\n"),
]


def _rewrite_known_macros(text: str) -> str:
    # Remove everything before \begin{document} (preamble)
    doc_start = text.find(r"\begin{document}")
    if doc_start != -1:
        text = text[doc_start:]

    # Remove LaTeX comment lines more safely:
    # - Strip only unescaped % (best-effort); avoid breaking things like \% in text.
    text = re.sub(r"(?m)(?<!\\)%.*$", "", text)

    for rx, rep in _LATEX_MACRO_REWRITES:
        text = rx.sub(rep, text)

    # Custom resume macros (template-specific)
    def handle_subheading(m):
        return f"\n{m.group(1)} | {m.group(2)}\n{m.group(3)} | {m.group(4)}\n"

    text = re.sub(
        r"\\resumeSubheading\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        handle_subheading,
        text
    )

    def handle_project(m):
        return f"\nProject: {m.group(1)}\nTools: {m.group(2)}\n"

    text = re.sub(
        r"\\resumeProject\{([^}]*)\}\{([^}]*)\}\{[^}]*\}\{[^}]*\}",
        handle_project,
        text
    )

    text = re.sub(
        r"\\resumePOR\{[^}]*\}\{([^}]*)\}\{([^}]*)\}",
        r"\1 (\2)\n",
        text
    )

    text = re.sub(r"\\resumeSubItem\{([^}]*)\}\{([^}]*)\}", r"\1: \2\n", text)

    # \item -> bullet
    text = re.sub(r"\\item\s*", "• ", text)

    return text


def _latex_to_text_with_parser(text: str) -> Optional[str]:
    """
    Try pylatexenc if installed for safer conversion.
    """
    try:
        from pylatexenc.latex2text import LatexNodes2Text  # type: ignore
        # Parse after macro rewrites for best results
        return LatexNodes2Text().latex_to_text(text)
    except Exception:
        return None


def _controlled_fallback_cleanup(text: str) -> str:
    """
    Regex fallback but less destructive than "wipe everything in []" etc.
    """
    # Remove environments but keep content best-effort (common CV envs)
    text = re.sub(r"\\begin\{(itemize|enumerate|tabular|center|flushleft|flushright)\}", "\n", text)
    text = re.sub(r"\\end\{(itemize|enumerate|tabular|center|flushleft|flushright)\}", "\n", text)

    # Remove remaining commands with one optional arg, but keep inner text when possible:
    # \cmd{...} -> ...
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", text)

    # Remove remaining bare commands: \cmd -> ''
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Drop stray braces
    text = re.sub(r"[{}]", "", text)

    # Keep bracket contents if it looks like real text; otherwise remove purely formatting-like brackets.
    # (Very conservative: only remove empty-ish bracket groups)
    text = re.sub(r"\[\s*\]", "", text)

    return text


def clean_latex(text: str) -> str:
    """
    Safer LaTeX-to-text:
    1) Rewrite known macros
    2) Try pylatexenc parser
    3) Fallback controlled cleanup
    4) Normalize whitespace
    """
    text = _rewrite_known_macros(text)

    parsed = _latex_to_text_with_parser(text)
    if parsed is not None:
        text = parsed
    else:
        text = _controlled_fallback_cleanup(text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# ---------------------------------------------------------------------------
# 2) Sectioning + Adaptive Chunking
# ---------------------------------------------------------------------------

def split_cv_sections(clean_text: str) -> list[Document]:
    """
    Splits CV text into sections, each becoming a separate Document.
    Uses headings formatted as: === TITLE ===
    """
    parts = re.split(r"=== (.+?) ===", clean_text)
    docs: list[Document] = []

    # Before first heading -> personal info
    if parts and parts[0].strip():
        docs.append(Document(page_content=parts[0].strip(), metadata={"section": "Personal Information"}))

    for i in range(1, len(parts) - 1, 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            docs.append(Document(page_content=f"{title}\n\n{content}", metadata={"section": title}))

    return docs


def normalize_section_name(raw: str) -> str:
    """
    Normalize section names into a smaller set for filtering/chunk policies.
    """
    s = (raw or "").lower()

    if any(k in s for k in ["experience", "deneyim", "work", "employment"]):
        return "Experience"
    if any(k in s for k in ["education", "eğitim", "university", "üniversite"]):
        return "Education"
    if any(k in s for k in ["project", "proje"]):
        return "Projects"
    if any(k in s for k in ["skill", "yetenek", "teknoloji", "tools", "stack"]):
        return "Skills"
    if any(k in s for k in ["cert", "sertifika", "certificate"]):
        return "Certifications"
    if any(k in s for k in ["personal", "information", "contact", "iletişim", "summary", "özet", "profil"]):
        return "Personal Information"
    return raw.strip() if raw else "Other"


def adaptive_split_documents(section_docs: list[Document]) -> list[Document]:
    """
    Applies section-based chunk settings:
    - Experience/Projects: larger chunks (more context)
    - Skills/Certifications: smaller chunks (more precision)
    - Personal: medium
    """
    chunks_all: list[Document] = []

    for d in section_docs:
        sec_raw = d.metadata.get("section", "Other")
        sec = normalize_section_name(sec_raw)

        if sec in ("Experience", "Projects"):
            chunk_size, overlap = 850, 120
            separators = ["\n\n", "\n", "• ", " - ", " "]
        elif sec in ("Skills", "Certifications"):
            chunk_size, overlap = 320, 60
            separators = ["\n", "• ", ", ", " "]
        elif sec in ("Education",):
            chunk_size, overlap = 550, 90
            separators = ["\n\n", "\n", "• ", " "]
        else:
            chunk_size, overlap = 450, 80
            separators = ["\n\n", "\n", "• ", " "]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
        )

        # Preserve normalized section name for filtering
        base = Document(
            page_content=d.page_content,
            metadata={
                **d.metadata,
                "section": sec,  # normalized
            }
        )

        chunks = splitter.split_documents([base])
        chunks_all.extend(chunks)

    # Add stable chunk_id + source metadata
    for i, c in enumerate(chunks_all, start=1):
        c.metadata["chunk_id"] = f"cv_chunk_{i:05d}"
        c.metadata["source"] = "CV.txt"

    return chunks_all


# ---------------------------------------------------------------------------
# 3) Load / Index (with CV hash control)
# ---------------------------------------------------------------------------

def load_cv(
    cv_path: str,
    collection_name: str = "furkan_cv",
    persist_directory: Optional[str] = "./chroma_furkan_cv"
) -> Chroma:
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"CV file not found: {cv_path}")

    cv_hash = file_sha256(cv_path)
    meta_path = os.path.join(persist_directory or ".", "cv_meta.json")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Try to reuse existing DB if hash matches
    can_reuse = False
    if persist_directory and os.path.isdir(persist_directory) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("cv_sha256") == cv_hash and meta.get("collection_name") == collection_name:
                can_reuse = True
        except Exception:
            can_reuse = False

    if can_reuse:
        vector_db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        print("✅ Mevcut ChromaDB bulundu, yeniden indeksleme yapılmadı.")
        print(f"📦 Persist directory: {os.path.abspath(persist_directory) if persist_directory else '(none)'}")
        return vector_db

    # Re-index
    with open(cv_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = clean_latex(raw_text)
    section_docs = split_cv_sections(clean_text)
    chunks = adaptive_split_documents(section_docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    if hasattr(vector_db, "persist"):
        try:
            vector_db.persist()
        except Exception:
            pass

    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"cv_sha256": cv_hash, "collection_name": collection_name},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception:
            pass

    print(f"✅ CV yüklendi: {len(chunks)} chunk eklendi.")
    print(f"📦 Persist directory: {os.path.abspath(persist_directory) if persist_directory else '(none)'}")
    return vector_db


# ---------------------------------------------------------------------------
# 4) RAG Tool (MMR + metadata filter + standardized evidence)
# ---------------------------------------------------------------------------

class CVSearchInput(BaseModel):
    query: str = Field(description="CV içinde aranacak soru/konu")
    section: Optional[str] = Field(default=None, description="(Opsiyonel) Section filtresi: Experience/Education/Projects/Skills/Certifications/Personal Information")
    k: int = Field(default=5, ge=1, le=10, description="Döndürülecek sonuç sayısı")


class CVSearchTool(BaseTool):
    name: str = "cv_search"
    description: str = (
        "Furkan KOÇAL'ın CV'sinde semantik arama yapar. "
        "Sonuçları standart kanıt formatında döndürür: chunk_id, section, source ve içerik."
    )
    args_schema: type[BaseModel] = CVSearchInput

    vector_db: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _mmr_search(self, query: str, k: int, section: Optional[str]) -> list[Document]:
        """
        Try MMR with metadata filter. Fallback to similarity_search.
        """
        if self.vector_db is None:
            return []

        # Normalize section (if given)
        sec = normalize_section_name(section) if section else None
        filter_dict = {"section": sec} if sec else None

        # Attempt MMR with filter
        try:
            if filter_dict is not None:
                return self.vector_db.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=max(12, k * 3),
                    lambda_mult=0.5,
                    filter=filter_dict,
                )
            return self.vector_db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=max(12, k * 3),
                lambda_mult=0.5,
            )
        except TypeError:
            # Some versions don't accept "filter" or args
            pass
        except Exception:
            pass

        # Fallback: similarity search (try with filter)
        try:
            if filter_dict is not None:
                return self.vector_db.similarity_search(query, k=k, filter=filter_dict)
        except Exception:
            pass

        try:
            return self.vector_db.similarity_search(query, k=k)
        except Exception:
            return []

    def _run(self, query: str, section: Optional[str] = None, k: int = 5, **kwargs) -> str:
        if self.vector_db is None:
            return json.dumps(
                {"ok": False, "error": "Vector DB bağlı değil. CVSearchTool(vector_db=...) ile oluştur."},
                ensure_ascii=False
            )

        docs = self._mmr_search(query=query, k=k, section=section)

        if not docs:
            return json.dumps(
                {
                    "ok": True,
                    "query": query,
                    "applied_section_filter": normalize_section_name(section) if section else None,
                    "results": [],
                },
                ensure_ascii=False
            )

        results = []
        for d in docs:
            sec = d.metadata.get("section", "")
            chunk_id = d.metadata.get("chunk_id", "")
            source = d.metadata.get("source", "CV.txt")
            content = (d.page_content or "").strip()

            # Keep excerpt short for quick display; full content also provided.
            excerpt = content
            if len(excerpt) > 360:
                excerpt = excerpt[:360].rstrip() + "…"

            results.append(
                {
                    "chunk_id": chunk_id,
                    "section": sec,
                    "source": source,
                    "excerpt": excerpt,
                    "content": content,
                }
            )

        return json.dumps(
            {
                "ok": True,
                "query": query,
                "applied_section_filter": normalize_section_name(section) if section else None,
                "results": results,
            },
            ensure_ascii=False
        )


# ---------------------------------------------------------------------------
# 5) UX: Default intro builder (ben kimim?)
# ---------------------------------------------------------------------------

def build_default_intro(cv_tool: CVSearchTool) -> dict:
    """
    CV'den güvenli bir 'kendini tanıt' özeti çıkarmak için çoklu sorgu yapar.
    Dönen yapı: {"facts": [...], "evidence": [...]}
    """
    queries = [
        ("Kişisel özet ve mevcut rol", "Personal Information"),
        ("Huawei deneyimi ve rol açıklaması", "Experience"),
        ("Eğitim bilgileri", "Education"),
        ("Öne çıkan projeler", "Projects"),
        ("Yetenekler ve teknolojiler", "Skills"),
        ("Sertifikalar", "Certifications"),
    ]

    evidence = []
    facts = []

    for q, sec in queries:
        raw = cv_tool._run(query=q, section=sec, k=3)
        try:
            payload = json.loads(raw)
        except Exception:
            continue

        for r in payload.get("results", [])[:2]:
            evidence.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "section": r.get("section"),
                    "excerpt": r.get("excerpt"),
                }
            )

        # Simple fact strings (still derived from CV text chunks)
        if payload.get("results"):
            # Use excerpts as fact candidates
            facts.append(
                {
                    "topic": q,
                    "section": sec,
                    "snippets": [x.get("excerpt") for x in payload["results"][:2]],
                }
            )

    return {"facts": facts, "evidence": evidence}


# ---------------------------------------------------------------------------
# 6) Agents (Prompt Safety tightened, Turkish consistent)
# ---------------------------------------------------------------------------

def create_agents(cv_search_tool: CVSearchTool):
    researcher = Agent(
        role="CV Araştırmacısı",
        goal=(
            "Kullanıcının sorusunu yanıtlamak için CV'den doğru kanıtları çıkar. "
            "Mutlaka cv_search aracını kullan. Varsayım yapma."
        ),
        backstory=(
            "Detaycı bir İK/teknik değerlendirme uzmanısın. Görevin sadece CV'den "
            "kanıt bulmak ve bunu yapılandırılmış şekilde dijital ikize vermek."
        ),
        tools=[cv_search_tool],
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    digital_twin = Agent(
        role="Furkan KOÇAL — Dijital İkiz",
        goal=(
            "Sadece araştırmacının sağladığı CV kanıtlarını kullanarak Türkçe cevap ver. "
            "CV dışına kesinlikle çıkma. Kanıt sunmadan iddia kurma."
        ),
        backstory=(
            "Sen Furkan KOÇAL'sın. Kullanıcı seninle sohbet ediyor. "
            "Ama kritik kural: yalnızca araştırmacının getirdiği CV kanıtlarına dayanabilirsin. "
            "CV'de yoksa dürüstçe 'CV'mde bu bilgi yer almıyor' dersin."
        ),
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, digital_twin


# ---------------------------------------------------------------------------
# 7) Question Runner (Intro UX + strict answer format)
# ---------------------------------------------------------------------------

def ask(question: str, researcher: Agent, digital_twin: Agent, cv_tool: CVSearchTool) -> str:
    # UX: intro questions -> create a synthetic "research packet" directly from CV
    if looks_like_intro_question(question):
        packet = build_default_intro(cv_tool)

        answer_task = Task(
            description=(
                f"Kullanıcının sorusu: '{question}'\n\n"
                "Aşağıdaki CV kanıt paketini kullanarak kendini Türkçe tanıt.\n\n"
                "KANIT PAKETİ (JSON):\n"
                f"{json.dumps(packet, ensure_ascii=False, indent=2)}\n\n"
                "KURALLAR (çok sıkı):\n"
                "1) Sadece bu paketteki kanıtlara dayan.\n"
                "2) CV'de olmayan hiçbir detayı ekleme.\n"
                "3) Sıcak ve doğal bir dille yaz ama CV okur gibi yapma.\n"
            ),
            expected_output=(
                "Türkçe, birinci tekil şahısla doğal tanıtım metni"
            ),
            agent=digital_twin,
        )

        crew = Crew(
            agents=[digital_twin],
            tasks=[answer_task],
            process=Process.sequential,
            verbose=True,
        )
        return str(crew.kickoff())

    # Normal flow: researcher -> twin
    # Researcher: use section heuristic to focus retrieval, but not mandatory
    sec_hint = infer_section_filter(question)

    research_task = Task(
        description=(
            f"Kullanıcının sorusu: '{question}'\n\n"
            "Mutlaka cv_search aracını kullan.\n"
            f"Eğer uygunsa section filtresi olarak şunu dene: {sec_hint!r} (uygun değilse boş bırak).\n\n"
            "Çıktıyı JSON şeklinde üret ve sadece CV kanıtlarından oluşsun:\n"
            "{\n"
            '  "answerable": true/false,\n'
            '  "facts": [\n'
            '     {"claim": "...", "chunk_id": "...", "section": "...", "evidence_excerpt": "..."}, ...\n'
            "  ],\n"
            '  "missing": ["CV\'de yoksa hangi alt bilgi eksik?" ...]\n'
            "}\n\n"
            "Kurallar:\n"
            "- En az 2 farklı chunk_id kullanmaya çalış.\n"
            "- Kanıt olmadan claim yazma.\n"
        ),
        expected_output="Yukarıdaki şemaya uygun JSON.",
        agent=researcher,
    )

    answer_task = Task(
        description=(
            f"Araştırmacının sağladığı JSON kanıtlara dayanarak şu soruyu Türkçe cevapla: '{question}'\n\n"
            "KURALLAR (çok sıkı):\n"
            "1) Sadece araştırmacı JSON'undaki 'facts' alanındaki bilgileri kullan.\n"
            "2) CV dışına çıkma; ek çıkarım yapma.\n"
            "3) Eğer 'answerable' false ise veya gerekli detay yoksa: 'CV'mde bu bilgi yer almıyor.' de.\n"
            "4) Sıcak ve doğal konuş ama kısa ve net ol.\n"
        ),
        expected_output="Türkçe cevap.",
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
