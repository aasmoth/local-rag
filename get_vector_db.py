import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# Ce fichier est conçu pour initialiser et retourner une instance 
# d'une base de données vectorielle Chroma, qui est utilisée pour 
# stocker et gérer des embeddings (représentations vectorielles) des textes. 
# Il fait appel à une méthode d'embedding basée sur le modèle de 
# texte défini, via la bibliothèque OllamaEmbeddings.

# chemin où les données de Chroma seront persistées. Par défaut, répertoire chroma (fichier .env)
CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
# nom de la collection dans la base de données Chroma. Par défaut, local-rag (fichier .env)
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
# nom du modèle d'embedding de texte. Par défaut, nomic-embed-text (fichier .env)
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')

def get_vector_db():
    # Initialisation de l'instance d'embedding Ollama
    # Crée une instance de la classe OllamaEmbeddings avec le modèle spécifié par TEXT_EMBEDDING_MODEL.
    # show_progress=True permet d'afficher la progression lors du calcul des embeddings.
    # OllamaEmbeddings est utilisé pour générer des représentations vectorielles des textes qui seront ensuite stockées dans Chroma.
    embedding = OllamaEmbeddings(model=TEXT_EMBEDDING_MODEL,show_progress=True)

    # Initialisation de l'instance de la base de données vectorielle Chroma
    db = Chroma(
        # Nom de la collection dans la base de données Chroma
        collection_name=COLLECTION_NAME,
        # Chemin où les données de Chroma seront persistées
        persist_directory=CHROMA_PATH,
        # Fonction d'embedding utilisée pour générer les embeddings des textes
        embedding_function=embedding
    )

    # Retourne l'instance de la base de données vectorielle Chroma utilisée dans embed.py pour stocker les chunks des documents
    return db