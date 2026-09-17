import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_gigachat import GigaChat
from pathlib import Path
from getpass import getpass

#---------------Настройка окружения и модели---------------#
ENV_DIR = Path("./data")
ENV_FILE = ENV_DIR / ".env"

def load_credentials():
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            lines = f.readlines()
        creds = {}
        for line in lines:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                creds[key] = value
        return creds.get("GIGACHAT_CLIENT_ID"), creds.get("GIGACHAT_CREDENTIALS"), creds.get("GIGACHAT_SCOPE")
    return None, None, None

def save_credentials(client_id: str, credentials: str, scope: str = "GIGACHAT_API_PERS"):
    ENV_DIR.mkdir(exist_ok=True)
    with open(ENV_FILE, "w") as f:
        f.write(f"GIGACHAT_CLIENT_ID={client_id}\n")
        f.write(f"GIGACHAT_CREDENTIALS={credentials}\n")
        f.write(f"GIGACHAT_SCOPE={scope}\n")
    print("Credentials saved successfully!")

CLIENT_ID, SECRET_KEY, SCOPE = load_credentials()
if not CLIENT_ID:
    print("No credentials found. Please enter your GigaChat access token.")
    CLIENT_ID = input("Your client id: ")
    SECRET_KEY = getpass("Access Token: ")   # скрытый ввод до кучи
    SCOPE = input("Scope (leave empty for GIGACHAT_API_PERS): ") or "GIGACHAT_API_PERS"
    save_credentials(CLIENT_ID, SECRET_KEY, SCOPE)
else:
    print("Credentials loaded from file.")

llm = GigaChat(
    credentials=SECRET_KEY,
    scope=SCOPE,
    verify_ssl_certs=False,
    model="GigaChat-2-Max"
)

#---------------Обрабатываем проекты---------------#
with open('./projects.json', 'r', encoding='utf-8') as f:
    projects = json.load(f)

project_texts = []
for p in projects:
    text = f"Название: {p['name']}. Описание: {p['description']}. Стек: {p['stack']}. Результат: {p['result']}."
    project_texts.append(text)

model = SentenceTransformer('deepvk/USER-bge-m3')
embeddings = model.encode(project_texts)

#---------------Поиск и ответ---------------#
def retrieve(query, top_k=2):
    query_emb = model.encode([query])
    sim = cosine_similarity(query_emb, embeddings)[0]
    best_indices = np.argsort(sim)[::-1][:top_k]
    return [projects[i] for i in best_indices], sim[best_indices]

def generate_answer(query, retrieved_projects):
    context = "\n\n".join([
        f"Проект: {p['name']}\nОписание: {p['description']}\nСтек: {p['stack']}\nРезультат: {p['result']}"
        for p in retrieved_projects
    ])
    prompt = f"""
    Ты — ассистент, который помогает рекрутерам узнать о проектах кандидата (твоего создателя).

    Найденные проекты:
    {context}

    Вопрос: {query}

    Ответь на вопрос, используя только информацию из проектов. Если в найденных проектах нет ответа, скажи, что не знаешь.
    """
    response = llm.invoke(prompt)
    return response.content

#---------------Основной цикл---------------#
print("Интерактивное резюме. Задай вопрос о проектах.")
while True:
    query = input("\nВопрос: ")
    if query.lower() in ['выход', 'exit', 'quit']:
        break
    projects_found, scores = retrieve(query)
    print("\nНайденные проекты:")
    for p, s in zip(projects_found, scores):
        print(f"  - {p['name']} (сходство: {s:.3f})")
    answer = generate_answer(query, projects_found)
    print("\nОтвет:", answer)