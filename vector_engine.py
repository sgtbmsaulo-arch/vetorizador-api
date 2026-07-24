import cv2
import numpy as np
from sklearn.cluster import KMeans
import svgwrite
import os

class NextGenVectorEngine:
    def __init__(self, image_path, max_colors=8, smooth_factor=0.003):
        """
        :param image_path: Caminho da imagem de entrada (PNG, JPG, etc.)
        :param max_colors: Número de cores para quantização e separação de camadas.
        :param smooth_factor: Fator de simplificação de curvas (Ramer-Douglas-Peucker).
        """
        self.image_path = image_path
        self.max_colors = max_colors
        self.smooth_factor = smooth_factor
        self.img_bgr = None
        self.img_rgb = None
        self.height = 0
        self.width = 0

    def load_and_preprocess(self, scale_factor=2.0):
        """Passo 1: Carregamento, Upscaling e Filtragem Bilateral de Ruído."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Imagem não encontrada: {self.image_path}")

        self.img_bgr = cv2.imread(self.image_path)
        
        # Upscaling suave para melhorar definição de bordas pequenas
        if scale_factor > 1.0:
            self.img_bgr = cv2.resize(
                self.img_bgr, (0, 0), 
                fx=scale_factor, fy=scale_factor, 
                interpolation=cv2.INTER_CUBIC
            )

        self.height, self.width, _ = self.img_bgr.shape

        # Filtro Bilateral: preserva bordas afiadas enquanto remove ruídos
        filtered = cv2.bilateralFilter(self.img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        self.img_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)

    def color_quantization(self):
        """Passo 2: Agrupamento de Cores por K-Means (Segmentação de Camadas)."""
        pixels = self.img_rgb.reshape((-1, 3))
        
        kmeans = KMeans(n_clusters=self.max_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        palette = np.uint8(kmeans.cluster_centers_)

        segmented_img = palette[labels].reshape((self.height, self.width, 3))
        labels_grid = labels.reshape((self.height, self.width))
        
        return palette, labels_grid

    def _contour_to_svg_path(self, contour):
        """Passo 3: Aplica algoritmo de simplificação e converte em caminho SVG."""
        epsilon = self.smooth_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            return None

        path_data = []
        start_point = approx[0][0]
        path_data.append(f"M {start_point[0]},{start_point[1]}")

        for pt in approx[1:]:
            p = pt[0]
            path_data.append(f"L {p[0]},{p[1]}")

        path_data.append("Z")
        return " ".join(path_data)

    def vectorize_to_svg(self, output_path="output_vector.svg"):
        """Passo 4: Processamento Topológico e Gravação da Arte Vetorial."""
        self.load_and_preprocess()
        palette, labels_grid = self.color_quantization()

        dwg = svgwrite.Drawing(
            output_path, 
            size=(self.width, self.height), 
            profile='full'
        )

        for i, color in enumerate(palette):
            hex_color = svgwrite.rgb(color[0], color[1], color[2], mode='RGB')
            color_mask = np.uint8(labels_grid == i) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

            contours, hierarchy = cv2.findContours(
                color_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            color_group = dwg.g(id=f"color_layer_{i}", fill=hex_color, stroke="none")

            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 15:
                    continue

                path_str = self._contour_to_svg_path(cnt)
                if path_str:
                    color_group.add(dwg.path(d=path_str))

            dwg.add(color_group)

        dwg.save()
