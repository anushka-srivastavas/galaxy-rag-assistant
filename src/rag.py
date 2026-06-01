from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import os

GALAXY_KNOWLEDGE = """
Disturbed Galaxy:
A disturbed galaxy shows signs of gravitational interaction with another galaxy. These galaxies have irregular shapes and asymmetric structures caused by tidal forces. They often show signs of recent or ongoing collisions. Examples include NGC 1427A. Disturbed galaxies are important for understanding galaxy evolution and merger processes.

Merging Galaxy:
Merging galaxies are two or more galaxies in the process of colliding and combining into a single galaxy. During a merger, tidal forces distort both galaxies, creating long tidal tails and bridges of stars. The Antennae Galaxies (NGC 4038/4039) are a famous example. Mergers trigger intense star formation and can fuel active galactic nuclei.

Round Smooth Galaxy:
Round smooth galaxies are elliptical galaxies with a nearly circular appearance. They contain mostly old, red stars and very little gas or dust. They have no spiral arms or disk structure. Giant elliptical galaxies like M87 are round smooth galaxies. They are commonly found at the centers of galaxy clusters.

In-between Round Smooth Galaxy:
These are lenticular or S0 galaxies that fall between elliptical and spiral galaxies. They have a disk structure like spiral galaxies but lack prominent spiral arms. They contain older stellar populations with little ongoing star formation. They are common in dense galaxy cluster environments.

Cigar Shaped Smooth Galaxy:
Cigar shaped smooth galaxies are elongated elliptical galaxies. They appear highly elongated due to their intrinsic shape or viewing angle. M82, also known as the Cigar Galaxy, is a famous example though it is actually a starburst galaxy. These galaxies contain old stellar populations.

Barred Spiral Galaxy:
Barred spiral galaxies have a central bar-shaped structure of stars from which spiral arms extend. About two thirds of all spiral galaxies including our Milky Way are barred spirals. The bar structure funnels gas toward the galactic center, triggering star formation. NGC 1300 is a classic example of a barred spiral galaxy.

Unbarred Tight Spiral Galaxy:
Unbarred tight spiral galaxies have tightly wound spiral arms without a central bar. They are classified as Sa type in the Hubble sequence. They have large central bulges and contain a mix of old and young stars. The spiral arms are well defined but closely spaced.

Unbarred Loose Spiral Galaxy:
Unbarred loose spiral galaxies have widely spaced, open spiral arms without a central bar. They are classified as Sc type in the Hubble sequence. They have small central bulges and are rich in gas and dust. They show active star formation in their spiral arms. M33, the Triangulum Galaxy, is an example.

Edge-on Galaxy without Bulge:
These are spiral galaxies viewed edge-on that lack a prominent central bulge. They appear as a thin flat disk of stars. The lack of bulge suggests they are late type spiral galaxies. Viewing them edge-on allows astronomers to study the vertical structure of galactic disks.

Edge-on Galaxy with Bulge:
These are spiral galaxies viewed edge-on with a prominent central bulge visible. They appear as a disk with a bright central concentration of stars. The bulge contains older stellar populations while the disk contains younger stars. The Sombrero Galaxy M104 is a famous example of an edge-on galaxy with a prominent bulge.
"""

def build_vectorstore():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([GALAXY_KNOWLEDGE])

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def get_answer(galaxy_class: str, vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.3
    )

    query = (
        f"Tell me everything about {galaxy_class}. "
        f"What are its characteristics, how does it form, "
        f"and what are famous examples?"
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Based on the following information about galaxies, answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content