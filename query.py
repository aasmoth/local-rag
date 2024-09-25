# Gère le processus de requête dans un système basé sur le modèle de langage sélectionné
# et la base de données vectorielle. Génère des questions alternatives pour améliorer 
# la recherche de documents pertinents dans une base de données vectorielle (Chroma) 
# et répond aux questions en utilisant uniquement le contexte extrait de la base de données.

import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from get_vector_db import get_vector_db



# variable d'environnement qui détermine le modèle de langage utilisé pour traiter 
# les requêtes. Par défaut, elle est définie sur mistral, mais elle peut être changée 
# en fonction des besoins
LLM_MODEL = os.getenv('LLM_MODEL', 'mistral')

# Fonction pour les templates de prompt
def get_prompt():
    # Un template qui génère cinq versions alternatives d'une question utilisateur pour aider à 
    # la recherche dans la base de données vectorielle. L'objectif est de surmonter certaines 
    # limitations des recherches basées uniquement sur la similarité de distance.
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Tu es un assistant de modèle linguistique d'IA. Ta tâche consiste à générer cinq
        versions différentes de la question donnée par l'utilisateur afin d'extraire les documents pertinents d'une base de données vectorielle. Toutes les réponses doivent être en français. Tu dois donner la référence du document en réponse.
        En générant des perspectives multiples sur la question de l'utilisateur, ton objectif est d'aider l'utilisateur à surmonter certaines des limites de la méthode basée sur la distance.
        votre objectif est d'aider l'utilisateur à surmonter certaines des limites de la recherche de similarité basée sur la distance.
        basée sur la distance. Tu fourniras ces questions alternatives séparées par des nouvelles lignes.
        Question initiale : {question}""",
    )

    template = """Réponds UNIQUEMENT aux questions basées sur le contexte suivant :
    {context}
    Question: {question}
    """

    # Ce template est utilisé pour répondre à la question en se basant uniquement sur 
    # le contexte récupéré (documents pertinents) de la base de données vectorielle.
    prompt = ChatPromptTemplate.from_template(template)

    return QUERY_PROMPT, prompt

# Fonction principale query
def query(input):
    if input:
        # Un modèle de langage ChatOllama est initialisé avec le modèle spécifié par la 
        # variable d'environnement LLM_MODEL (ici, par défaut mistral)
        llm = ChatOllama(model=LLM_MODEL)
        # Accès à la base vectorielle. La fonction get_vector_db() est utilisée pour obtenir 
        # une instance de la base de données vectorielle (Chroma) précédemment définie
        db = get_vector_db()
        # Récupération du prompt de template
        QUERY_PROMPT, prompt = get_prompt()

        # Cette partie du code configure un retriever qui génère plusieurs versions de la 
        # question d'entrée à l'aide du modèle de langage (llm) et du template QUERY_PROMPT.
        # Le but est de récupérer des documents pertinents dans la base de données en utilisant 
        # ces différentes versions de la question.
        retriever = MultiQueryRetriever.from_llm(
            db.as_retriever(), 
            llm,
            prompt=QUERY_PROMPT
        )

        # Chaine de traitement pour laquelle un pipeline est défini :
        # 1. Contexte récupéré à partir de la donnée via le retriever
        # 2. Question originale transmise sans modification via RunnablePassthrough
        # 3. Réponse basée sur le contexte
        # 4. Parsing : La sortie est analysée via StrOutputParser() pour obtenir un résultat formaté en texte
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Si la réponse est fournie, la réponse finale est calculée et retournée, sinon None
        response = chain.invoke(input)
        
        return response

    return None