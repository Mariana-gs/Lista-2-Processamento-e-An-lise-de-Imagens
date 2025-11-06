# Relatório de Segmentação e Análise de Forma: K-means, Otsu e Fecho Convexo

**Autora: Mariana Galvão Soares**

Este repositório contém o código-fonte desenvolvido para o trabalho da disciplina de Processamento e Análise de Imagens.

O objetivo deste projeto é implementar e comparar duas técnicas de segmentação de imagens (K-Means e Limiarização de Otsu) e aplicar uma técnica de representação de forma (Fecho Convexo).

## 📖 Sobre o Script

O script `main.py` automatiza o seguinte fluxo para um conjunto de imagens de entrada:

1. **Segmentação por K-Means:** Aplica o algoritmo K-Means (com `K=5`) para segmentar a imagem com base na similaridade de cor.
2.  **Segmentação por Otsu:** Converte a imagem para escala de cinza e aplica o método de Otsu para encontrar um limiar global automático, gerando uma máscara binária.
3.  **Geração de Histograma:** Salva o histograma da imagem em escala de cinza.
4.  **Representação de Forma:** Encontra o maior contorno na máscara de Otsu (invertida)  e calcula o **Fecho Convexo** sobre ele.
5.  **Salvamento:** Salva todas as imagens de resultado (K-Means, Otsu, Fecho Convexo) e uma imagem comparativa final na pasta `resultados/`.

## 🛠️ Instalação e Dependências

Recomenda-se o uso de um ambiente virtual (virtual environment) para instalar as dependências.

1.  Crie e ative um ambiente virtual:
    ```sh
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Instale as bibliotecas necessárias:
    ```sh
    pip install -r requirements.txt
    ```
