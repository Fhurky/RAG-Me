"""
Dijital İkiz CV Sistemi — Furkan KOÇAL
---------------------------------------
LaTeX .txt formatındaki CV'yi okur, temizler ve CrewAI + ChromaDB ile
soru-cevap sistemi kurar.

Kurulum:
    pip install crewai langchain langchain-community chromadb sentence-transformers
"""

import re
import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from crewai import LLM


# ---------------------------------------------------------------------------
# 1. ADIM: LaTeX Temizleyici
# ---------------------------------------------------------------------------

def latex_temizle(metin: str) -> str:
    """LaTeX komutlarını metinden temizler, saf içeriği döndürür."""

    # \begin{document} öncesini (preamble) sil
    doc_baslangic = metin.find(r"\begin{document}")
    if doc_baslangic != -1:
        metin = metin[doc_baslangic:]

    # Yorum satırlarını sil
    metin = re.sub(r"%.*", "", metin)

    # \href{url}{metin} → metin
    metin = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", metin)

    # \textbf{}, \textit{} vb. → içerik
    metin = re.sub(r"\\text(?:bf|it|rm|sf|tt|sc|up|sl)\{([^}]*)\}", r"\1", metin)
    metin = re.sub(r"\\(?:small|large|Large|huge|Huge|normalsize)\{([^}]*)\}", r"\1", metin)
    metin = re.sub(r"\\(?:footnotesize|scriptsize)\{([^}]*)\}", r"\1", metin)

    # \section{} → başlık
    metin = re.sub(r"\\section\{\\textbf\{([^}]*)\}\}", r"\n=== \1 ===\n", metin)
    metin = re.sub(r"\\section\{([^}]*)\}", r"\n=== \1 ===\n", metin)

    # \resumeSubheading{kurum}{yer}{rol}{tarih}
    def subheading_isle(m):
        return f"\n{m.group(1)} | {m.group(2)}\n{m.group(3)} | {m.group(4)}\n"
    metin = re.sub(
        r"\\resumeSubheading\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}",
        subheading_isle, metin
    )

    # \resumeProject{başlık}{araçlar}{...}{...}
    def project_isle(m):
        return f"\nProje: {m.group(1)}\nAraçlar: {m.group(2)}\n"
    metin = re.sub(
        r"\\resumeProject\{([^}]*)\}\{([^}]*)\}\{[^}]*\}\{[^}]*\}",
        project_isle, metin
    )

    # \resumePOR{}{içerik}{tarih}
    metin = re.sub(
        r"\\resumePOR\{[^}]*\}\{([^}]*)\}\{([^}]*)\}",
        r"\1 (\2)\n", metin
    )

    # \resumeSubItem{başlık}{içerik}
    metin = re.sub(r"\\resumeSubItem\{([^}]*)\}\{([^}]*)\}", r"\1: \2\n", metin)

    # \item
    metin = re.sub(r"\\item\s*", "• ", metin)

    # Kalan LaTeX komutları
    metin = re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", "", metin)
    metin = re.sub(r"\\[a-zA-Z]+\*?", "", metin)

    # Parantezleri temizle
    metin = re.sub(r"[{}]", "", metin)
    metin = re.sub(r"\[.*?\]", "", metin)

    # Çoklu boş satır → tek satır
    metin = re.sub(r"\n{3,}", "\n\n", metin)
    metin = "\n".join(line.strip() for line in metin.splitlines())

    return metin.strip()


def cv_bolumle(temiz_metin: str) -> list:
    """CV metnini bölümlere ayırır, her bölüm ayrı Document olur."""
    bolumler = re.split(r"=== (.+?) ===", temiz_metin)
    docs = []

    # Başlık öncesi kısım (kişisel bilgiler)
    if bolumler[0].strip():
        docs.append(Document(
            page_content=bolumler[0].strip(),
            metadata={"section": "Kişisel Bilgiler"}
        ))

    for i in range(1, len(bolumler) - 1, 2):
        baslik = bolumler[i].strip()
        icerik = bolumler[i + 1].strip() if i + 1 < len(bolumler) else ""
        if icerik:
            docs.append(Document(
                page_content=f"{baslik}\n\n{icerik}",
                metadata={"section": baslik}
            ))

    return docs


def cv_yukle(cv_yolu: str = "cv.txt") -> Chroma:
    """CV .txt dosyasını okur, temizler ve ChromaDB'ye yükler."""
    if not os.path.exists(cv_yolu):
        raise FileNotFoundError(f"CV dosyası bulunamadı: {cv_yolu}")

    with open(cv_yolu, "r", encoding="utf-8") as f:
        ham_metin = f.read()

    temiz_metin = latex_temizle(ham_metin)
    bolumler = cv_bolumle(temiz_metin)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", "• ", " "]
    )
    parcalar = splitter.split_documents(bolumler)

    # Ücretsiz, Türkçe+İngilizce destekli embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vektor_db = Chroma.from_documents(
        documents=parcalar,
        embedding=embeddings,
        collection_name="furkan_cv"
    )

    print(f"✅ CV yüklendi: {len(parcalar)} parça ChromaDB'ye eklendi.")
    return vektor_db


# ---------------------------------------------------------------------------
# 2. ADIM: RAG Aracı
# ---------------------------------------------------------------------------

class CVAramaInput(BaseModel):
    sorgu: str = Field(description="CV'de aranacak konu veya soru")


class CVAramaTool(BaseTool):
    name: str = "cv_arama"
    description: str = (
        "Furkan KOÇAL'ın CV'sinde semantik arama yapar. "
        "Deneyim, eğitim, projeler, beceriler veya sertifikalar hakkında "
        "bilgi almak için bu aracı kullan."
    )
    args_schema: type[BaseModel] = CVAramaInput
    vektor_db: object = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, sorgu: str) -> str:
        sonuclar = self.vektor_db.similarity_search(sorgu, k=4)
        if not sonuclar:
            return "CV'de bu konuyla ilgili bilgi bulunamadı."
        parcalar = []
        for doc in sonuclar:
            bolum = doc.metadata.get("section", "")
            parcalar.append(f"[{bolum}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parcalar)


# ---------------------------------------------------------------------------
# 3. ADIM: Ajanlar
# ---------------------------------------------------------------------------

ollama_llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

def ajanlar_olustur(cv_arama_tool: CVAramaTool):

    arastirmaci = Agent(
        role="CV Araştırmacısı",
        goal=(
            "Kullanıcının sorusuna en uygun CV bilgilerini bulmak. "
            "Her zaman cv_arama aracını kullanarak somut, doğrulanmış veri getir."
        ),
        backstory=(
            "Sen deneyimli bir İK uzmanısın. Furkan KOÇAL'ın CV'sini detaylıca "
            "analiz eder, sorularla örtüşen deneyim, proje ve becerileri bulursun."
        ),
        tools=[cv_arama_tool],
        verbose=True,
        allow_delegation=False,
    )

    dijital_ikiz = Agent(
        role="Furkan KOÇAL — Dijital İkiz",
        goal=(
            "Araştırmacının bulduğu CV verilerini kullanarak soruya "
            "Furkan'ın kendisi olarak, 1. tekil şahısla ve Türkçe cevap vermek."
        ),
        backstory=(
            "Sen Furkan KOÇAL'sın — Yıldız Teknik Üniversitesi Bilgisayar Mühendisliği "
            "mezunu, şu an Huawei'de AI Research Engineer olarak çalışıyorsun. "
            "Yapay zeka, makine öğrenmesi ve derin öğrenme alanlarında tutkulusun. "
            "Profesyonel, samimi ve özgüvenlisin. "
            "Asla CV'de olmayan bir bilgiyi uydurmaz, "
            "bilmediğin şeyleri nazikçe kabul edersin."
        ),
        verbose=True,
        allow_delegation=False,
    )

    return arastirmaci, dijital_ikiz


# ---------------------------------------------------------------------------
# 4. ADIM: Soru Çalıştırıcı
# ---------------------------------------------------------------------------

def sor(soru: str, arastirmaci: Agent, dijital_ikiz: Agent) -> str:

    gorev_arastir = Task(
        description=(
            f"Kullanıcının sorusu: '{soru}'\n\n"
            "cv_arama aracını kullanarak bu soruyla ilgili Furkan'ın CV'sindeki "
            "tüm bilgileri topla. Bulduklarını özet halinde sun."
        ),
        expected_output="CV'den toplanan ilgili bilgilerin özeti (bölüm adlarıyla).",
        agent=arastirmaci,
    )

    gorev_cevapla = Task(
        description=(
            f"Araştırmacının getirdiği CV bilgilerini kullanarak "
            f"'{soru}' sorusuna Furkan KOÇAL olarak cevap ver.\n\n"
            "Kurallar:\n"
            "• Türkçe yaz\n"
            "• 1. tekil şahıs kullan ('Ben...', 'Çalışıyorum...', 'Geliştirdim...')\n"
            "• Sadece araştırmacının getirdiği bilgilere dayan\n"
            "• CV'de olmayan bilgileri uydurma, 'Bu konuda CV'mde bilgi yok' de\n"
            "• Samimi ve profesyonel bir ton kullan"
        ),
        expected_output="Furkan'ın 1. tekil şahısla verdiği Türkçe, doğal cevap.",
        agent=dijital_ikiz,
        context=[gorev_arastir],
    )

    ekip = Crew(
        agents=[arastirmaci, dijital_ikiz],
        tasks=[gorev_arastir, gorev_cevapla],
        process=Process.sequential,
        verbose=True,
    )

    return str(ekip.kickoff())


# ---------------------------------------------------------------------------
# 5. ADIM: Ana Döngü
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CV_YOLU = "1771864439371_CV.txt"  # cv dosya adını buraya yaz

    vektor_db = cv_yukle(CV_YOLU)
    cv_tool = CVAramaTool(vektor_db=vektor_db)
    arastirmaci, dijital_ikiz = ajanlar_olustur(cv_tool)

    print("\n🤖 Furkan'ın Dijital İkizi hazır! Çıkmak için 'q' yaz.\n")
    print("Örnek sorular:")
    print("  - Hangi programlama dillerini biliyorsun?")
    print("  - Huawei'deki rolünden bahseder misin?")
    print("  - Projelerinde hangi teknolojileri kullandın?\n")

    while True:
        soru = input("Sorunuz: ").strip()
        if soru.lower() in ("q", "quit", "çıkış", "exit"):
            print("Görüşmek üzere!")
            break
        if not soru:
            continue

        print("\n" + "=" * 60)
        cevap = sor(soru, arastirmaci, dijital_ikiz)
        print(f"\n💬 Furkan:\n{cevap}")
        print("=" * 60 + "\n")