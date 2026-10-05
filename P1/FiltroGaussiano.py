import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d


def construir_imagen():

    fondo_negro = np.zeros((256, 256), dtype = np.float64)
    fondo = fondo_negro + 0.15

    cuadrado = np.zeros_like(fondo_negro)
    cuadrado[64:192, 64:192] = 0.30


    circulo = np.zeros_like(fondo_negro)

    y, x = np.ogrid[:256, :256]
    mascara = (x - 127.5)**2 + (y - 127.5)**2 <= 32**2
    circulo[mascara] =  0.35

    return fondo + cuadrado + circulo


def kernel_gauss(desv_est):

    medio_largo = np.ceil(2 * desv_est)

    y, x = np.ogrid[-medio_largo:medio_largo + 1 , -medio_largo:medio_largo + 1]

    kernel = np.exp(-(x**2 + y**2)/(2 * desv_est **2))

    return kernel/np.sum(kernel)


def convolucion(kernel: np.ndarray, imagen: np.ndarray):

    return convolve2d(imagen, kernel, mode="same", boundary="symm")


def rmse(imagen_ruido: np.ndarray, imagen_original: np.ndarray, sigmas: np.ndarray):

    cuad = np.zeros((256, 256), dtype=bool)
    cuad[64:192, 64:192] = True

    y, x = np.ogrid[:256, :256]        
    circ = (x - 127.5)**2 + (y - 127.5)**2 <= 32**2

    mascaras_dict = {"global": np.ones_like(imagen_ruido, dtype=bool), "fondo": ~cuad, "cuadrado": cuad & ~circ , "circulo": circ}

    resultados_dict = {k: [] for k in mascaras_dict}

    for s in sigmas:
        if s == 0:
            filtrada = imagen_ruido
        else:
            kernel = kernel_gauss(s)
            filtrada = convolucion(kernel, imagen_ruido)

        for llave , masc in mascaras_dict.items():
            diff = filtrada[masc] - imagen_original[masc]
            resultados_dict[llave].append(np.sqrt(np.mean(diff**2)))

    return resultados_dict


def graficar_rmse(sigmas, resultados):
    
    fig, ax = plt.subplots(figsize=(7, 5))

    colores = {"fondo": "tab:blue", "cuadrado": "tab:orange",
                   "circulo": "tab:green", "global": "k"}

    for nombre, valores_s in resultados.items():
        valores_s = np.asarray(valores_s)
        es_global = (nombre == "global")
        color = colores.get(nombre, None)

        ax.plot(sigmas, valores_s, label=nombre, color=color, linewidth=1.6, linestyle="--" if es_global else "-")

        i_min = np.argmin(valores_s)

        ax.plot(sigmas[i_min], valores_s[i_min], "o", color=color, zorder=5)
        ax.annotate(f"desv_est*={sigmas[i_min]:.2f}", (sigmas[i_min], valores_s[i_min]),
                    textcoords="offset points", xytext=(5, 6),
                    fontsize=8, color=color)

    ax.set_xlabel("Desviación Estandar")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE(σ) por región")

    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()

    return fig


def interpolacion_sigma(imagen_suavizada: np.ndarray, puntos_control):

    mu = puntos_control[:,0]
    sigmas = puntos_control[:,1]

    return np.interp(imagen_suavizada, mu, sigmas )


def graficar_F_mu(mu_puntos, sigma_puntos):

    fig, ax = plt.subplots(figsize=(6, 4))

    mu_malla = np.linspace(0, 1, 300)
    sigma_de_mu = np.interp(mu_malla, mu_puntos, sigma_puntos)

    ax.plot(mu_malla, sigma_de_mu, color="tab:blue", label="F(μ̂) interpolada")
    ax.plot(mu_puntos, sigma_puntos, "o", color="red", zorder=5,
            label="puntos de control (σ óptimos)")

    nombres = ["fondo", "cuadrado", "círculo"]
    for mu, s, nombre in zip(mu_puntos, sigma_puntos, nombres):
        ax.annotate(nombre, (mu, s), textcoords="offset points",
                    xytext=(6, 6), fontsize=9, color="red")

    ax.set_xlabel("μ̂")
    ax.set_ylabel("σ")
    ax.set_title("Funcion σ = F(μ̂)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig


def filtro_gaussiano_adaptativo(imagen_ruidosa, mapa_sigma):

    imagen_filtrada = np.zeros_like(imagen_ruidosa)
    
    sigma_maximo = np.max(mapa_sigma)
    radio_maximo = int(np.ceil(2 * sigma_maximo))
    
    img_pad = np.pad(imagen_ruidosa, pad_width=radio_maximo, mode='edge')
    
    for i in range(256):
        for j in range(256):
            
            sigma_local = mapa_sigma[i, j]
            kernel = kernel_gauss(sigma_local)
            
            radio = int(np.ceil(2 * sigma_local))

            i_pad = i + radio_maximo
            j_pad = j + radio_maximo

            vecindad = img_pad[i_pad - radio : i_pad + radio + 1, 
                               j_pad - radio : j_pad + radio + 1]
            

            imagen_filtrada[i, j] = np.sum(vecindad * kernel)
            
    return imagen_filtrada

def comparar_filtros_imagen(imagen_global, imagen_adaptativo):

    diff = imagen_global - imagen_adaptativo


    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    
    axs[0].imshow(imagen_global, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Global σ=1.6")

    axs[1].imshow(imagen_adaptativo, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Adaptativo σ_suavizado = 1.9")

    im = axs[2].imshow(diff, cmap="RdBu_r", vmin=-0.02, vmax=0.02)
    axs[2].set_title("Diferencia (global - adaptativo)")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    

    plt.tight_layout()

    return fig


rng = np.random.default_rng(23)


imagen_original = construir_imagen()
imagen_ruido = rng.poisson(imagen_original * 40)/40

kernel_global, kernel_referencia = kernel_gauss(1.6), kernel_gauss(1.9)

imagen_filtro = convolucion(kernel_global, imagen_ruido)

sigmas = np.arange(0, 6.05, 0.1)
resultados = rmse(imagen_ruido, imagen_original, sigmas)

imagen_suavizada = convolucion(kernel_referencia, imagen_ruido)
puntos_control = np.array([(0.15, 1.8), (0.45, 1.4), (0.80, 1.5)])


mapa_sigma = interpolacion_sigma(imagen_suavizada, puntos_control)


imagen_filtro_adaptado = filtro_gaussiano_adaptativo(imagen_ruido, mapa_sigma)

'''Para poder visualizar correctamente las diferentes graficas, hay que descomentar la requerida y comentar las demás.'''
# =============================================================================================================

'''Grafica de una unica imagen'''
#plt.imshow(imagen_filtro_adaptado, cmap='gray', vmin=0.0, vmax=1.0)

'''Grafica de curvas RMSE | Valores entre 0 y 6 |'''
#graficar_rmse(sigmas, resultados)

'''Grafica del mapeo de σ(x, y)'''
#plt.imshow(interpolacion_sigma(imagen_suavizada, puntos_control), cmap="viridis")

'''Grafica de la curva de σ = F(μ̂) '''
#graficar_F_mu(puntos_control[:,0], puntos_control[:,1])

'''Grafica de comparación de flitros y su diferencia'''
#comparar_filtros_imagen(imagen_filtro, imagen_filtro_adaptado)
#================================================================================================================

plt.show()

