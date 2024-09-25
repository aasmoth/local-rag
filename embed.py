import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_vector_db import get_vector_db

TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')

# Vérifie si le fichier est autorisé (uniquement PDF)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

# Sauvegarde en répertoire temporaire
def save_file(file):
    # Génération du nom de fichier avec un horodatage et sauvegarde dans répertoire temporaire
    ct = datetime.now()
    ts = ct.timestamp()
    filename = str(ts) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TEMP_FOLDER, filename)
    file.save(file_path)

    return file_path

# Chargement et découpage du PDF
def load_and_split_data(file_path):
    # Découpe le texte en morceaux de 7500 caractères avec un chevauchement de 100 caractères
    # Permet de mieux structurer les documents pour l'analyse vectorielle
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    return chunks

# Processus d'embedding principal
def embed(file):
    # Si le fichier est valide, on le sauvegarde, on le découpe et on l'ajoute à la base 
    # de données grâce à get_vector_db()
    if file.filename != '' and file and allowed_file(file.filename):
        file_path = save_file(file)
        chunks = load_and_split_data(file_path)
        db = get_vector_db()
        db.add_documents(chunks)
        db.persist()
        os.remove(file_path)

        return True

    return False