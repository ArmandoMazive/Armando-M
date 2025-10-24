import os
import json
import getpass
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from modelNeural import NeuralRecommender
#  Funções de autenticação

USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def login():
    users = load_users()
    print("\n=== LOGIN ===")
    username = input("Usuário: ")
    password = getpass.getpass("Senha: ")

    if username in users and users[username] == password:
        print(f"\nBem-vindo de volta, {username}!\n")
        return username
    else:
        print("\nUsuário ou senha incorretos!\n")
        return None

def register():
    users = load_users()
    print("\n=== CADASTRO ===")
    username = input("Novo usuário: ")
    if username in users:
        print("Usuário já existe!")
        return None
    password = getpass.getpass("Crie uma senha: ")
    users[username] = password
    save_users(users)
    print(f"Conta criada com sucesso para {username}!\n")
    return username

#  Funções do recomendador

def carregar_filmes():
    return pd.read_csv("movies.csv")

def preparar_modelo(filmes):
    textos = filmes["title"] + " " + filmes["genre"] + " " + filmes["description"]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(textos)

    model = NeuralRecommender(X.shape[1])
    with torch.no_grad():
        embeddings = model(torch.tensor(X.toarray(), dtype=torch.float32))
    return filmes, embeddings

def recomendar_filme(filme_input, filmes, embeddings):
    idx = filmes[filmes["title"].str.lower() == filme_input.lower()].index
    if len(idx) == 0:
        print(" Filme não encontrado!\n")
        return

    idx = idx[0]
    input_vector = embeddings[idx].unsqueeze(0)
    sims = torch.matmul(embeddings, input_vector.T).squeeze(1)
    filmes["similaridade"] = sims.numpy()

    recomendacoes = filmes.sort_values(by="similaridade", ascending=False)
    recomendacoes = recomendacoes[recomendacoes["title"] != filme_input].head(3)

    print(f"\n🎥 Já que você gostou de '{filme_input}', talvez goste de:")
    for _, row in recomendacoes.iterrows():
        print(f"👉 {row['title']} - {row['genre']}")
    print()


def main():
    print("=== SISTEMA DE RECOMENDAÇÃO NEURAL ===")

    user = None
    while not user:
        opcao = input("\n1 - Login\n2 - Cadastrar\nEscolha: ")
        if opcao == "1":
            user = login()
        elif opcao == "2":
            user = register()
        else:
            print("Opção inválida!")

    filmes = carregar_filmes()
    filmes, embeddings = preparar_modelo(filmes)

    while True:
        filme = input("\nDigite o nome de um filme (ou 'sair' para terminar): ")
        if filme.lower() == "sair":
            print("Até logo!")
            break
        recomendar_filme(filme, filmes, embeddings)

if __name__ == "__main__":
    main()
