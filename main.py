# Processamento e Análise de Imagens - Lista 2
# Algoritmos: K-Means e Limiarização de Otsu com Fecho Convexo
# Mariana Galvão Soares

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Definição das pastas
INPUT_DIR = "imagens"
OUTPUT_DIR = "resultados"

# 2. Lista de imagens a processar
IMAGE_NAMES = ["arara.png", "urso.jpg", "dioneia.jpg"]

# 3. Parâmetro para o K-Means
K_VALUE = 5


def salvar_histograma(image_gray, base_filename, thresh_val=None):
    """
    Calcula e salva o histograma da imagem em escala de cinza.
    (Versão modificada para NÃO desenhar a linha do limiar).
    """
    # 1. Calcular o histograma
    hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

    # 2. Configurar o plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Histograma - {base_filename}", fontsize=16)
    plt.xlabel("Nível de Intensidade (0-255)", fontsize=12)
    plt.ylabel("Contagem de Pixels", fontsize=12)
    plt.plot(hist, color='black')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, 256])
   
    # 3. Salvar a figura
    output_path = os.path.join(OUTPUT_DIR, f"histograma_{base_filename}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Histograma salvo em: {output_path}")


def load_image_safely(path):
    """
    Carrega uma imagem usando um método que suporta nomes de arquivo 
    com caracteres non-ASCII (como 'ç', 'ã', etc.).
    """
    try:
        # Lê o arquivo como bytes e decodifica com o OpenCV
        n = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Não foi possível carregar a imagem em: {path}")
        return img
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}")
        return None

def segment_kmeans(image, K):
    """
    Segmenta a imagem usando K-Means no espaço de cor L*a*b*.
    O espaço L*a*b* é perceptualmente mais uniforme que o RGB,
    o que melhora os resultados do clustering.
    """
    # 1. Converter para o espaço de cor L*a*b*
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 2. Achatar a imagem em uma lista de pixels (amostras x 3 canais)
    # e converter para float32, exigido pelo cv2.kmeans
    pixels = lab_image.reshape((-1, 3)).astype(np.float32)

    # 3. Definir critérios e aplicar K-Means
    # Critérios: parar após 10 iterações ou se a precisão (epsilon) for 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Tentar 10 inicializações diferentes e pegar o melhor resultado
    compactness, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 4. Mapear os pixels de volta para as cores do centroide
    # 'centers' são as K cores L*a*b* encontradas
    segmented_colors = centers[labels.flatten()]

    # 5. Remodelar a imagem de volta ao formato original (H, W, C)
    segmented_image_lab = segmented_colors.reshape(image.shape)

    # 6. Converter de float32 de volta para uint8 (tipo de imagem)
    # e depois converter do espaço L*a*b* de volta para BGR para salvar
    segmented_image_bgr = cv2.cvtColor(segmented_image_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    return segmented_image_bgr


def segment_otsu(image, base_filename): 
    """
    Segmenta a imagem usando o limiar (threshold) global de Otsu
    e salva seu histograma.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar o threshold de Otsu PRIMEIRO para obter o valor
    thresh_val, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"O limiar de Otsu para '{base_filename}' foi: {thresh_val}")
    
    salvar_histograma(gray, base_filename, thresh_val) # Passa o valor aqui
    
    return binary_image

def aplicar_fecho_convexo(binary_mask, original_image_to_draw_on):
    """
    Encontra o fecho convexo na máscara binária e o desenha na imagem original.
    INVERTE a máscara antes de processar, como solicitado.
    """
    # 1. INVERTER A MÁSCARA 
    # Otsu fez o objeto de interesse ser PRETO (0)
    # e o fundo BRANCO (255). findContours procura objetos BRANCOS.
    inverted_mask = cv2.bitwise_not(binary_mask)

    # 2. Encontrar os contornos na máscara invertida
    # cv2.RETR_EXTERNAL pega apenas os contornos externos 
    # cv2.CHAIN_APPROX_SIMPLE economiza memória ao armazenar os pontos do contorno
    contours, hierarchy = cv2.findContours(inverted_mask, 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma cópia da imagem original para desenhar
    hull_image = original_image_to_draw_on.copy()

    # 3. Verificar se algum contorno foi encontrado
    if len(contours) > 0:

        # 4. Encontrar o maior contorno 
        largest_contour = max(contours, key=cv2.contourArea)

        # 5. Calcular o Fecho Convexo 
        hull = cv2.convexHull(largest_contour)

        # 6. Desenhar o fecho convexo na imagem
        cv2.drawContours(hull_image, [hull], -1, (0, 255, 0), 3)

    return hull_image

def display_and_save_results(original, kmeans_result, otsu_result, hull_result, base_filename):
    """
    Mostra e salva um gráfico comparativo dos resultados (4 imagens).
    """
    # 1. Criar a figura para o plot (largura aumentada para 4 imagens)
    plt.figure(figsize=(28, 7)) 

    # 2. Plot: Imagem Original
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original", fontsize=16)
    plt.axis('off')

    # 3. Plot: K-Means
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(kmeans_result, cv2.COLOR_BGR2RGB))
    plt.title(f"K-Means (K={K_VALUE})", fontsize=16)
    plt.axis('off')

    # 4. Plot: Otsu
    plt.subplot(1, 4, 3)
    plt.imshow(otsu_result, cmap='gray')
    plt.title("Limiarização de Otsu", fontsize=16)
    plt.axis('off')
    
    # 5. Plot: Fecho Convexo (NOVO)
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(hull_result, cv2.COLOR_BGR2RGB))
    plt.title("Fecho Convexo (Otsu Invertido)", fontsize=16)
    plt.axis('off')

    # 6. Salvar o gráfico comparativo
    output_path_comparison = os.path.join(OUTPUT_DIR, f"comparacao_{base_filename}.png")
    plt.savefig(output_path_comparison)
    print(f"Gráfico comparativo salvo em: {output_path_comparison}")
    plt.close()

    # 8. Salvar os arquivos de resultado individuais
    output_path_kmeans = os.path.join(OUTPUT_DIR, f"kmeans_{base_filename}.png")
    output_path_otsu = os.path.join(OUTPUT_DIR, f"otsu_{base_filename}.png")
    output_path_hull = os.path.join(OUTPUT_DIR, f"hull_{base_filename}.png")

    _, buf_kmeans = cv2.imencode(".png", kmeans_result)
    buf_kmeans.tofile(output_path_kmeans)
    
    _, buf_otsu = cv2.imencode(".png", otsu_result)
    buf_otsu.tofile(output_path_otsu)
    
    _, buf_hull = cv2.imencode(".png", hull_result)
    buf_hull.tofile(output_path_hull)

# --- Função Principal ---

def main():
    # Criar a pasta de resultados se ela não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Pasta criada: {OUTPUT_DIR}")

    # Iterar sobre todas as imagens
    for img_name in IMAGE_NAMES:
        # Construir o caminho completo do arquivo
        img_path = os.path.join(INPUT_DIR, img_name)
        
        # Obter o nome do arquivo sem extensão (para salvar os resultados)
        base_filename = os.path.splitext(img_name)[0]
        
        print(f"\n--- Processando: {img_name} ---")

        # 1. Carregar a imagem
        original_image = load_image_safely(img_path)
        if original_image is None:
            continue # Pula para a próxima imagem se esta falhou

        # 2. Aplicar Algoritmo 1: K-Means
        print("Aplicando K-Means...")
        kmeans_result = segment_kmeans(original_image, K_VALUE)

        # 3. Aplicar Algoritmo 2: Otsu
        print("Aplicando Otsu e gerando histograma...")
        otsu_result = segment_otsu(original_image, base_filename)
        
        # 4. Aplicar o Fecho Convexo
        print("Aplicando Fecho Convexo...")
        hull_result = aplicar_fecho_convexo(otsu_result, original_image)

        # 5. Salvar e mostrar resultados
        print("Salvando resultados...")
        display_and_save_results(original_image, kmeans_result, otsu_result, hull_result, base_filename)

    print("\n--- Processamento Concluído ---")
    print(f"Todos os resultados foram salvos na pasta: {OUTPUT_DIR}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()