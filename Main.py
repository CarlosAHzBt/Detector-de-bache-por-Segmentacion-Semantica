import os
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from skimage.measure import label, regionprops
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from Segformer_FineTuner import SegformerFinetuner

class ModeloSegmentacion:
    def __init__(self, modelo_entrenado):
        self.min_area = 3000  # Área mínima para considerar una detección de bache válida
        self.modelo = modelo_entrenado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preparar_imagen(self, ruta_imagen):
        imagen = Image.open(ruta_imagen).convert("RGB")
        transformaciones = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        imagen_transformada = transformaciones(imagen).unsqueeze(0).to(self.device)
        return imagen_transformada

    def _aplicar_modelo(self, pixel_values):
        if torch.cuda.is_available():
            pixel_values = pixel_values.to('cuda')
        with torch.no_grad():
            predicciones = self.modelo(pixel_values=pixel_values)
            predicted_mask = (predicciones[0].argmax(dim=1) == 1).squeeze().cpu().numpy().astype(int)
        return predicted_mask

    def _redimensionar_mascara(self, predicted_mask):
        predicted_mask_resized = resize(predicted_mask, (480, 848), order=0, 
                                        preserve_range=True, anti_aliasing=False).astype(int)
        return predicted_mask_resized

    def _etiquetar_regiones(self, mask_resized):
        labeled_mask = label(mask_resized, connectivity=2)
        return labeled_mask

    def _filtrar_regiones(self, labeled_mask):
        regions = regionprops(labeled_mask)
        filtered_regions = [region.coords for region in regions if region.area >= self.min_area]
        return filtered_regions
    
    def _dibujar_regiones_filtradas(self, labeled_mask, filtered_regions):
        plt.imshow(labeled_mask, cmap="nipy_spectral")
        for region in filtered_regions:
            plt.plot(region[:, 1], region[:, 0], "o", markersize=3)
        plt.show()

    def obtener_coordenadas_baches(self, ruta_imagen):
        pixel_values = self._preparar_imagen(ruta_imagen)
        predicted_mask = self._aplicar_modelo(pixel_values)
        mask_resized = self._redimensionar_mascara(predicted_mask)
        labeled_mask = self._etiquetar_regiones(mask_resized)
        coordenadas_baches = self._filtrar_regiones(labeled_mask)
        self._dibujar_regiones_filtradas(labeled_mask, coordenadas_baches) #Opcional,(Debuging) para ver mascaras de segmentacion
        return coordenadas_baches

class CargarModelo:
    def cargar_modelo(self, ruta_modelo):
        id2label = {
            0: "background",
            1: "Bache",
        }
        modelo = SegformerFinetuner(id2label=id2label)  # Asegúrate de proporcionar los argumentos necesarios aquí
        modelo.load_state_dict(torch.load(ruta_modelo))
        modelo.eval()
        return modelo

def procesar_imagenes(carpeta_imagenes, ruta_modelo, carpeta_resultados):
    cargador_modelo = CargarModelo()
    modelo_segmentacion = ModeloSegmentacion(cargador_modelo.cargar_modelo(ruta_modelo))

    if not os.path.exists(carpeta_resultados):
        os.makedirs(carpeta_resultados)

    for nombre_imagen in os.listdir(carpeta_imagenes):
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_imagen)
        if ruta_imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
            imagen = Image.open(ruta_imagen).convert("RGB")
            pixel_values = modelo_segmentacion._preparar_imagen(ruta_imagen)
            predicted_mask = modelo_segmentacion._aplicar_modelo(pixel_values)
            mask_resized = modelo_segmentacion._redimensionar_mascara(predicted_mask)
            
            # Guardar la imagen original, la máscara y la combinación
            imagen.save(os.path.join(carpeta_resultados, f"{nombre_imagen}_original.png"))
            plt.imsave(os.path.join(carpeta_resultados, f"{nombre_imagen}_mask.png"), mask_resized, cmap="jet")

            # Dibujar detecciones sobre la imagen
            plt.figure()
            plt.imshow(imagen)
            plt.imshow(mask_resized, alpha=0.5, cmap="nipy_spectral")
            plt.axis('off')
            plt.savefig(os.path.join(carpeta_resultados, f"{nombre_imagen}_deteccion.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    carpeta_imagenes = "imagenes"
    ruta_modelo = "model_state_dictV18-este_ya_trae_ruido.pth"
    carpeta_resultados = "resultados"

    procesar_imagenes(carpeta_imagenes, ruta_modelo, carpeta_resultados)
