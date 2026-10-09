import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import cv2

'''Funciones utiles de la P1'''
#============================================================================================
def kernel_gauss(desv_est):

    medio_largo = np.ceil(2 * desv_est)

    y, x = np.ogrid[-medio_largo:medio_largo + 1 , -medio_largo:medio_largo + 1]

    kernel = np.exp(-(x**2 + y**2)/(2 * desv_est **2))

    return kernel/np.sum(kernel)


def convolucion(kernel: np.ndarray, imagen: np.ndarray):

    return convolve2d(imagen, kernel, mode="same", boundary="symm")
#=============================================================================================

def ruido_gauss(imagen_gris: np.ndarray):

    Alto, Ancho = imagen_gris.shape

    rng = np.random.default_rng(23)
    ruido = rng.normal(scale=0.05, size=(Alto, Ancho))

    imagen_ruido = imagen_gris + ruido

    return imagen_ruido


def difusion_anisotropica(imagen_ruido: np.ndarray, funcion_c, lambd: float, iteraciones: int, **parametros_c ):  

    I = imagen_ruido.copy().astype(np.float64)

    for i in range(iteraciones):

        i_pad = np.pad(I, 1, mode="edge") 

        d_n = i_pad[:-2, 1:-1] - I         
        d_s = i_pad[2:, 1:-1]  - I         
        d_e = i_pad[1:-1, 2:]  - I          
        d_o = i_pad[1:-1, :-2] - I 

        c_n = funcion_c(np.abs(d_n), I, **parametros_c)
        c_s = funcion_c(np.abs(d_s), I, **parametros_c)
        c_e = funcion_c(np.abs(d_e), I, **parametros_c)
        c_o = funcion_c(np.abs(d_o), I, **parametros_c)

        I = I + lambd*(c_n*d_n + c_s*d_s + c_e*d_e + c_o*d_o)

    return I


def c_tv(diff, I, epsilon):

    a = 1.0 / (np.sqrt(diff**2 + epsilon**2))

    return a

def aplicar_laplaciano(imagen):

    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)

    return convolucion(kernel, imagen)

def funcion_c_laplaciano(diff, I, gamma, epsilon, desv_est=1.0):

    imagen_suavizada = convolucion(kernel_gauss(desv_est), I)

    imagen_laplaciano = np.abs(aplicar_laplaciano(imagen_suavizada))

    escala_gradiente = np.std(diff)
    escala_laplaciano = np.std(imagen_laplaciano)

    E = (diff/escala_gradiente) + gamma * (imagen_laplaciano/escala_laplaciano)

    return 1.0/(1.0 + (E/epsilon)**2)


def funcion_c_gradiente(diff, I, epsilon):

    return np.maximum(0.0, 1.0 - (np.abs(diff) / epsilon)**2) ** 2


def graficar_mapa_c(imagen, funcion_c, **parametros_c):

    I = imagen.copy().astype(np.float64)
    i_pad = np.pad(I, 1, mode="edge") 

    d_n = i_pad[:-2, 1:-1] - I         
    d_s = i_pad[2:, 1:-1]  - I         
    d_e = i_pad[1:-1, 2:]  - I          
    d_o = i_pad[1:-1, :-2] - I 

    c_n = funcion_c(np.abs(d_n), I, **parametros_c)
    c_s = funcion_c(np.abs(d_s), I, **parametros_c)
    c_e = funcion_c(np.abs(d_e), I, **parametros_c)
    c_o = funcion_c(np.abs(d_o), I, **parametros_c)

    c_dict ={"N": c_n, "S": c_s, "E": c_e, "W": c_o}


    fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
    for ax, (nombre, c) in zip(axs, c_dict.items()):
        im = ax.imshow(c, cmap="viridis", vmin=0)
        ax.set_title(f"$c_{{{nombre}}}$"); ax.axis("off")
    fig.suptitle("Mapa de c")
    fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02)

    return fig, c_dict

def rmse_imagenes(imagen_a, imagen_b, mascara=None):

    diff = imagen_a.astype(np.float64) - imagen_b.astype(np.float64)
    if mascara is not None:
        diff = diff[mascara]
    return np.sqrt(np.mean(diff**2))


imagen = cv2.imread("Imagenes/gato.jfif")
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris = imagen_gris.astype(np.float64) / 255.0
imagen_ruido = ruido_gauss(imagen_gris)
imagen_dif_an_tv = difusion_anisotropica(imagen_ruido, c_tv, lambd = 1.25, iteraciones=50, epsilon= 0.0)
imagen_dif_an_lap = difusion_anisotropica(imagen_ruido, funcion_c_laplaciano, lambd=0.125, iteraciones=50, gamma=0.5, epsilon=1.0)
imagen_dif_an_grad = difusion_anisotropica(imagen_ruido, funcion_c_gradiente, lambd=0.125, iteraciones=50, epsilon=1.0)


'''Para poder visualizar correctamente las diferentes graficas, hay que descomentar la requerida y comentar las demás.'''
#====================================================================================================================================

'''Grafica comparativa de la imagen ruidosa y los tres filtros aplicados'''

'''fig, axs = plt.subplots(1, 4, figsize=(16,6))
axs[0].imshow(imagen_ruido, cmap="gray", vmin=0, vmax=1); axs[0].set_title("Ruidosa")
axs[1].imshow(imagen_dif_an_tv, cmap="gray", vmin=0, vmax=1); axs[1].set_title("Filtro TV")
axs[2].imshow(imagen_dif_an_lap, cmap="gray", vmin=0, vmax=1); axs[2].set_title("Filtro Laplaciano")
axs[3].imshow(imagen_dif_an_grad, cmap="gray", vmin=0, vmax=1); axs[3].set_title("Filtro Propuesto 2")'''

'''Grafica del mapa del coeficiente de difusión c para un filtro específico'''
#graficar_mapa_c(imagen_ruido, c_tv, epsilon=5.0)

'''Grafica de una unica imagen resultante'''
plt.imshow(imagen_dif_an_tv, cmap="gray", vmin=0, vmax=1)

#=====================================================================================================================================


plt.show()


resultados = {
    "Ruidosa":      imagen_ruido,
    "TV":           imagen_dif_an_tv,
    "Propuesta 1 (Laplaciano)":  imagen_dif_an_lap,
    "Propuesta 2":  imagen_dif_an_grad
}

for nombre, img in resultados.items():
    print(f"{nombre}: RMSE = {rmse_imagenes(img, imagen_gris):.5f}")