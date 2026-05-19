import csv
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import community as community_louvain
from collections import defaultdict
import matplotlib.cm as cm
from infomap import Infomap
import scipy as sp
import random
from collections import defaultdict, Counter
from networkx.drawing import nx_agraph
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, average
import igraph as ig
import math
from itertools import combinations
from networkx.drawing.nx_agraph import graphviz_layout

# --------------------------------------------------------------------------------
# CURRÍCULUM: cronología como cadenas por bloque (sin enlaces entre bloques)
# --------------------------------------------------------------------------------
import unicodedata
import re

def _norm_txt(s: str) -> str:
    """Normaliza texto para emparejar ítems del currículum con nodos del grafo."""
    if s is None:
        return ""
    s = str(s).replace("...", " ").replace("…", " ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # quitar tildes
    s = s.lower()
    s = re.sub(r"[^a-z0-9\sñç]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _match_curriculum_item_to_node(curr_item: str, existing_nodes: list[str]) -> str | None:
    ni = _norm_txt(curr_item)
    if not ni:
        return None

    # 1) Match exacto normalizado
    norm_map = {_norm_txt(n): n for n in existing_nodes}
    if ni in norm_map:
        return norm_map[ni]

    # 2) Match por similitud de palabras (Jaccard) usando todo el texto
    words_i = set(ni.split())
    if not words_i:
        return None

    best_node = None
    best_score = -1.0

    for n in existing_nodes:
        nn = _norm_txt(n)
        words_n = set(nn.split())
        if not words_n:
            continue

        inter = len(words_i & words_n)
        union = len(words_i | words_n)
        score = inter / union if union else 0.0

        if score > best_score:
            best_score = score
            best_node = n

    # umbral: ajusta si hace falta
    if best_score < 0.30:
        return None
    return best_node

def construir_curr_pos_desde_cronologia(G_base: nx.DiGraph):
    """
    Devuelve curr_pos: {nodo_en_tu_red: (bloque, orden_en_bloque)}
    según la cronología del currículum (textos completos).
    """
    faltantes = []

    bloques = {
        1: [
            "Determinación, a través del cálculo vectorial, del campo gravitatorio producido por un sistema de masas. Efectos sobre las variables cinemáticas y dinámicas de objetos inmersos en el campo.",
            "Momento angular de un objeto en un campo gravitatorio: cálculo, relación con las fuerzas centrales y aplicación de su conservación en el estudio de su movimiento.",
            "Energía mecánica de un objeto sometido a un campo gravitatorio: deducción del tipo de movimiento que posee, cálculo del trabajo o los balances energéticos existentes en desplazamientos entre diferentes posiciones, velocidades y tipos de trayectorias.",
            "Leyes que se verifican en el movimiento planetario y extrapolación al movimiento de satélites y cuerpos celestes.",
        ],
        2: [
            "Campos eléctrico y magnético: tratamiento vectorial, determinación de las variables cinemáticas y dinámicas de cargas eléctricas libres en presencia de estos campos. Fenómenos naturales y aplicaciones tecnológicas en los cuales se aprecian estos efectos.",
            "Intensidad del campo eléctrico en distribuciones de cargas discretas y continuas: cálculo e interpretación del flujo de campo eléctrico.",
            "Energía de una distribución de cargas estáticas: magnitudes que se modifican y que permanecen constantes como el desplazamiento de cargas libres entre puntos de diferente potencial eléctrico.",
            "Campos magnéticos generados por hilos con corriente eléctrica en diferentes configuraciones geométricas: rectilíneos, espiras, solenoides o toros. Interacción con cargas eléctricas libres presentes a su entorno.",
            "Líneas de campo eléctrico y magnético producidas por distribuciones de carga sencillas, imanes e hilos con corriente eléctrica en diferentes configuraciones geométricas.",
            "Determinación de variables cinemáticas y dinámicas de las cargas en campos eléctricos y magnéticos: ley de Lorentz.",
            "Variación de flujo magnético. Generación de la fuerza electromotriz: funcionamiento de motores, generadores y transformadores a partir de sistemas donde se produce una variación del flujo magnético.",
            "El campo magnético y su relación con el campo eléctrico.",
        ],
        3: [
            # 2.4.1 Movimientos oscilatorios
            "Determinación de las variables cinemáticas de un movimiento oscilatorio.",
            "La conservación de la energía mecánica.",
            "Análisis de gráficas de oscilación.",
            "El movimiento armónico simple.",
            # 2.4.2 Definición de fenómenos ondulatorios
            "¿Qué es un fenómeno ondulatorio?",
            "El concepto de onda mecánica. Tipos de ondas mecánicas.",
            "Identificación en la naturaleza y aplicaciones.",
            "¿Qué es el sonido? Tratamiento del sonido como fenómeno ondulatorio.",
            "Cualidades de las ondas sonoras. Atenuación y umbral sonoro.",
            "Contaminación acústica y otras aplicaciones.",
            "Situaciones y contextos naturales en los cuales se ponen de manifiesto diferentes fenómenos ondulatorios. Interferencias y difracción. Aplicaciones. Cambios en las propiedades de las ondas en función del desplazamiento del emisor y receptor.",
            # 2.4.3 La naturaleza de la luz
            "La luz ligada a la visión. La cámara oscura.",
            "La descomposición en colores en un prisma.",
            "La luz como onda electromagnética.",
            "El experimento de la doble rendija.",
            # 2.4.4 Espectro electromagnético
            "El espectro visible.",
            "El descubrimiento del infrarrojo: El espectro no visible.",
            "Características de estas ondas: frecuencia y longitud de onda.",
            "Diferencias con las ondas mecánicas.",
            "Esquema del espectro electromagnético, presencia en el entorno tecnológico y escala comparativa.",
            # 2.4.5 Óptica geométrica
            "Índice de refracción.",
            "Formación de imágenes en medios y objetos con diferente índice de refracción. Sistemas ópticos: lentes, prismas, espejos planos y curvos.",
            "Aplicaciones de la Óptica geométrica.",
        ],
        4: [
            # 2.5.1 Relatividad especial
            "Principios fundamentales de la relatividad especial.",
            "Dilatación del tiempo y contracción de la longitud.",
            "Equivalencia masa-energía. Energía y masa relativistas.",
            "Implicaciones en el cambio de paradigma de la mecánica clásica.",
            # 2.5.2 Carácter cuántico
            "Concepto de cuanto: hipótesis de Max Planck.",
            "Descripción del efecto fotoeléctrico en términos de paquetes de energía. El concepto de fotón.",
            "Hipótesis de De Broglie.",
            "Controversias históricas originadas por la naturaleza de la materia y la energía, derivadas de la dualidad onda-corpúsculo en la luz.",
            "El principio de incertidumbre formulado para el tiempo y la energía.",
            "Papel de la física cuántica en aplicaciones como el láser, resonancias magnéticas o nanotecnología.",
            # 2.5.3 Física nuclear y partículas
            "La radiactividad natural y otros procesos nucleares.",
            "Núcleos atómicos y estabilidad de isótopos.",
            "Modelo estándar de la física de partículas.",
            "Aceleradores de partículas.",
            "Clasificación de las partículas elementales.",
            "Interacciones fundamentales como intercambio de partículas (bosones).",
            "Fisión y fusión nuclear.",
            "Otras aplicaciones en los campos de la ingeniería, la tecnología y la salud.",
        ],
    }

    existing_nodes = list(G_base.nodes())
    curr_pos = {}

    for b in sorted(bloques.keys()):
        for k, item in enumerate(bloques[b]):
            nodo = _match_curriculum_item_to_node(item, existing_nodes)
            if nodo is None:
                faltantes.append((b, k, item))
                print(f"[Aviso] Ítem del currículum NO encontrado en tu red: {item[:90]}...")
                continue
            # si un nodo se mapea varias veces, nos quedamos con la primera vez (más temprano)
            curr_pos.setdefault(nodo, (b, k))

    print(f"\n[Resumen currículum] Emparejados: {len(curr_pos)} | No emparejados: {len(faltantes)}")
    for (b, k, item) in faltantes:
        print(f" - Bloque {b}, pos {k}: {item}")

    return curr_pos



def construir_grafo_curriculum_cadenas_sin_interbloques(G_base: nx.DiGraph):
    """
    Construye el grafo del currículum como 4 cadenas (una por bloque),
    SIN enlaces entre bloques.
    """
    curr_pos = construir_curr_pos_desde_cronologia(G_base)

    # agrupar por bloque y ordenar por k
    bloques = {}
    for nodo, (b, k) in curr_pos.items():
        bloques.setdefault(b, []).append((k, nodo))
    for b in bloques:
        bloques[b].sort(key=lambda x: x[0])
        bloques[b] = [n for _, n in bloques[b]]

    G_curr = nx.DiGraph()
    G_curr.add_nodes_from(curr_pos.keys())

    # cadenas internas por bloque: u->v sucesivos
    for b in sorted(bloques.keys()):
        nodos_b = bloques[b]
        for i in range(len(nodos_b) - 1):
            G_curr.add_edge(nodos_b[i], nodos_b[i + 1])

    return G_curr, curr_pos

def comparar_dag_vs_curriculum(G_dag: nx.DiGraph, curr_pos: dict):
    """Cuenta cuántas aristas del DAG violan el orden del currículum (por bloque y orden)."""
    comparables = 0
    violaciones = []

    for u, v in G_dag.edges():
        if u in curr_pos and v in curr_pos:
            comparables += 1
            if curr_pos[u] >= curr_pos[v]:
                violaciones.append((u, v, curr_pos[u], curr_pos[v]))

    ratio = (len(violaciones) / comparables) if comparables else None
    return comparables, violaciones, ratio

def dibujar_grafo_curriculum(G_curr: nx.DiGraph, curr_pos: dict, titulo="Grafo del currículum (cadenas por bloque, sin enlaces entre bloques)"):
    """Dibuja el grafo del currículum en 4 columnas (bloques)."""
    pos = {n: (b, -k) for n, (b, k) in curr_pos.items()}

    plt.figure(figsize=(14, 9))
    nx.draw_networkx_nodes(G_curr, pos, node_size=260, alpha=0.95)
    nx.draw_networkx_edges(G_curr, pos, arrows=True, width=1.2, alpha=0.7)
    nx.draw_networkx_labels(G_curr, pos, font_size=6)
    plt.title(titulo)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Solicitar la ruta del archivo CSV antes de ejecutar
archivo_csv = r"C:\Users\pablo\Desktop\PAPERS\Redes\redfisica.csv"

def cargar_grafo_desde_csv(ruta_csv):
    """Carga un grafo mixto desde un CSV con nodos dirigidos y no dirigidos."""
    if not os.path.exists(ruta_csv):
        print(f" Error: El archivo {ruta_csv} no existe.")
        return None

    with open(ruta_csv, 'r', newline='', encoding='utf-8') as file:
        reader = list(csv.reader(file))

    reader = [fila for fila in reader if any(fila)]  # 🔹 Eliminar filas vacías
    nombres_nodos = reader[0]  # 🔹 Extraer nombres de nodos

    G = nx.DiGraph()        # red dirigida
    Gno = nx.Graph()        # red no dirigida
    Gtotal = nx.DiGraph()   # red conjunta (dirigida)
    Gtotalsimple = nx.Graph()  # red conjunta simple (sin duplicar enlaces)

    # Añadir nodos
    for nombre in nombres_nodos[1:]:  # Desde la segunda columna
        G.add_node(nombre)
        Gno.add_node(nombre)
        Gtotal.add_node(nombre)
        Gtotalsimple.add_node(nombre)

    # Procesar matriz de adyacencia
    for i, fila in enumerate(reader[1:], start=1):  # Desde la segunda fila
        nodo_origen = nombres_nodos[i]  # Nombre del nodo según cabecera
        for j, valor in enumerate(fila[1:], start=1):  # Desde la segunda columna
            nodo_destino = nombres_nodos[j]
            if nodo_origen == nodo_destino or valor == '':  # 🔹 Evitar autoconexiones y celdas vacías
                continue

            try:
                valor = int(valor)
            except ValueError:
                continue  # Si no es un entero, se ignora

            if valor == 0:
                continue  # 0 indica que no hay relación

            if valor == 1:
                G.add_edge(nodo_origen, nodo_destino, tipo="dirigido")  # A → B
                Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")
            elif valor == 2:
                G.add_edge(nodo_destino, nodo_origen, tipo="dirigido")  # B → A
                Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_destino, nodo_origen, tipo="simple")
            elif valor == 3:
                Gno.add_edge(nodo_origen, nodo_destino, tipo="no dirigido")
                # En la red total lo metes como dos dirigidos (ida y vuelta)
                Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")

    return G, Gno, Gtotal, Gtotalsimple


def clean_edge_labels(ax):
    """
    Elimina los cuadros blancos detrás de las etiquetas de aristas en un gráfico Matplotlib.
    """
    for artist in ax.get_children():
        if isinstance(artist, plt.Text):
            # Desactiva el fondo de las etiquetas de texto
            artist.set_bbox(dict(facecolor='none', edgecolor='none'))


def filtrar_red_por_grado(grafo, umbral_grado):
    """
    Devuelve un subgrafo con los nodos cuyo grado (in-degree + out-degree) es mayor al umbral.
    """
    grados = dict(grafo.degree())
    nodos_filtrados = [n for n, g in grados.items() if g > umbral_grado]
    return grafo.subgraph(nodos_filtrados).copy()


def jerarquia_nodos(grafo):
    """
    Devuelve un diccionario con la jerarquía de los nodos en el DAG,
    basada en la longitud de los caminos más largos para cada nodo.
    """
    if not nx.is_directed_acyclic_graph(grafo):
        raise ValueError("El grafo debe ser un DAG para calcular la jerarquía de nodos.")

    # Calcula la jerarquía como la longitud máxima de cualquier camino hacia el nodo
    jerarquia = {}
    for nodo in nx.topological_sort(grafo):
        # Si el nodo no tiene predecesores, su jerarquía es 0
        predecesores = list(grafo.predecessors(nodo))
        if not predecesores:
            jerarquia[nodo] = 0
        else:
            jerarquia[nodo] = 1 + max(jerarquia[p] for p in predecesores)

    return jerarquia



# --------------------------------------------------------------------------------
# 1. Crear el grafo dirigido (ejemplo o cargado desde un archivo)
# --------------------------------------------------------------------------------


def get_analysis_subgraph(G_original, n=10):
    """
    Obtiene un subgrafo a partir de los n nodos con mayor grado en G_original.
    """
    degree_dict = dict(G_original.degree())
    # Ordenar nodos por grado de forma descendente
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:n]
    # Crear subgrafo inducido
    subgraph = G_original.subgraph(top_nodes).copy()
    return subgraph
    """
    Obtiene un subgrafo a partir de los n nodos con mayor grado en G_original.
    """
    degree_dict = dict(G_original.degree())
    # Ordenar nodos por grado de forma descendente
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:n]
    # Crear subgrafo inducido
    subgraph = G_original.subgraph(top_nodes).copy()
    return subgraph

def calcular_metricas_e_infomap(G):
    """
    Calcula métricas de centralidad sobre el grafo G y lo exporta para Infomap.
    Además genera diferentes grafos de la red.

    Parámetros:
        G (nx.DiGraph o nx.Graph): Grafo de NetworkX.

    """
    # Se asegura de que sea un di-grafo para Infomap, aunque se puede adaptar a grafo simple si se requiere
    if not isinstance(G, (nx.DiGraph, nx.Graph)):
        raise TypeError("G debe ser un grafo dirigido (nx.DiGraph) o no dirigido (nx.Graph).")

    # --------------------------
    # Cálculo de métricas
    # --------------------------

    # Centralidad de grado
    grado_centrality = nx.degree_centrality(G)
    # Betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    # Closeness centrality
    closeness_centrality = nx.closeness_centrality(G)

    # PáginaRank (solo tiene sentido en grafos dirigidos)
    if isinstance(G, nx.DiGraph):
        pagerank_centrality = nx.pagerank(G)
    else:
        pagerank_centrality = None

    # Coeficiente de clustering
    clustering_centrality = nx.clustering(G.to_undirected() if isinstance(G, nx.DiGraph) else G)
    # Vector de eigenvector centrality (para grafos no dirigidos)
    if not isinstance(G, nx.DiGraph):
        eigenvector_centrality = nx.eigenvector_centrality(G)
    else:
        eigenvector_centrality = None  # Se puede implementar para grafos dirigidos también

    # --------------------------
    # Exportar a Infomap
    # --------------------------

    # Nombre del archivo sin extensión
    filename_base = "red_network"

    # Guardar el grafo en formato Pajek
    pajek_path = filename_base + ".net"
    nx.write_pajek(G, pajek_path)
    print(f"Grafo exportado a {pajek_path} para Infomap.")

    # --------------------------
    # Crear archivos auxiliares
    # --------------------------

    # Crear un DataFrame con todas las métricas para facilitar análisis
    data = {
        'node': list(G.nodes()),
        'degree': [G.degree(n) for n in G.nodes()],
        'betweenness': [betweenness_centrality[n] for n in G.nodes()],
        'closeness': [closeness_centrality[n] for n in G.nodes()],
        'clustering': [clustering_centrality[n] for n in G.nodes()],
    }

    if pagerank_centrality is not None:
        data['pagerank'] = [pagerank_centrality[n] for n in G.nodes()]

    if eigenvector_centrality is not None:
        data['eigenvector'] = [eigenvector_centrality[n] for n in G.nodes()]

    df = pd.DataFrame(data)
    df_path = filename_base + "_metrics.csv"
    df.to_csv(df_path, index=False)
    print(f"Métricas de centralidad guardadas en {df_path}.")

    # --------------------------
    # Hacer particiones con Infomap
    # --------------------------

    im = Infomap()
    for u, v in G.edges():
        im.add_link(u, v)
    im.run()

    # Obtener comunidades
    communities = defaultdict(list)
    for node in im.nodes:
        communities[node.module_id].append(node.node_id)

    # Guardar comunidades en un archivo
    communities_path = filename_base + "_communities.txt"
    with open(communities_path, "w") as f:
        for module_id, nodes in communities.items():
            f.write(f"Community {module_id}: {', '.join(map(str, nodes))}\n")
    print(f"Comunidades detectadas con Infomap guardadas en {communities_path}.")

    # --------------------------
    # Visualizar el grafo
    # --------------------------
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout para el grafo

    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='blue', alpha=0.7)
    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.5)
    # Etiquetas opcionales
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Visualización del Grafo con sus Comunidades (Infomap)")
    plt.axis('off')
    plt.show()

# --------------------------------------------------------------------------------
# Herramientas auxiliares
# --------------------------------------------------------------------------------

def get_top_nodes_by_centrality(G, metric='degree', top_n=10):
    """
    Obtiene los top_n nodos por la métrica de centralidad especificada.

    Parámetros:
        G (nx.Graph o nx.DiGraph): Grafo de NetworkX.
        metric (str): Métrica de centralidad a usar. Opciones:
                      'degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector'.
        top_n (int): Número de nodos a devolver.

    Retorna:
        list of tuples: Lista de (nodo, valor_métrica) ordenada de mayor a menor.
    """

    centrality = None

    if metric == 'degree':
        centrality = dict(G.degree())
    elif metric == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif metric == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif metric == 'pagerank':
        if isinstance(G, nx.DiGraph):
            centrality = nx.pagerank(G)
        else:
            raise ValueError("PageRank solo está definido para grafos dirigidos.")
    elif metric == 'eigenvector':
        if not isinstance(G, nx.DiGraph):
            centrality = nx.eigenvector_centrality(G)
        else:
            raise ValueError("Eigenvector centrality se implementa aquí solo para grafos no dirigidos.")
    else:
        raise ValueError(f"Métrica de centralidad '{metric}' no reconocida.")

    # Ordenar y devolver los top_n
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_nodes

def main():
    grafodirigido, grafonodirigido, grafototal, grafototalsimple = cargar_grafo_desde_csv(archivo_csv)

    if grafototal is None:
        return

    print("Número de nodos en la red total:", nx.number_of_nodes(grafodirigido))
    print("Número de enlaces en la red total:", nx.number_of_edges(grafodirigido))

    # ----------------------------------------------------------------
    # 1b) COMPONENTES FUERTEMENTE CONEXAS (SCC) Y NODOS / ENLACES CÍCLICOS
    # ----------------------------------------------------------------
    # Usamos la red dirigida original (grafodirigido), antes de romper ciclos.
    sccs = list(nx.strongly_connected_components(grafodirigido))

    nodos_en_scc_ciclicas = set()
    for comp in sccs:
        if len(comp) > 1:
            # Cualquier SCC con más de un nodo es necesariamente cíclica.
            nodos_en_scc_ciclicas.update(comp)
        elif len(comp) == 1:
            # Caso especial: un solo nodo con bucle (self-loop) también es cíclico.
            nodo = next(iter(comp))
            if grafodirigido.has_edge(nodo, nodo):
                nodos_en_scc_ciclicas.add(nodo)

    num_nodos_totales_dir = grafodirigido.number_of_nodes()
    num_nodos_ciclicos = len(nodos_en_scc_ciclicas)
    cociente_nodos_ciclicos = (num_nodos_ciclicos / num_nodos_totales_dir) if num_nodos_totales_dir > 0 else 0.0

    # Ahora calculamos qué parte de los ENLACES dirigidos caen dentro de SCCs cíclicas.
    num_enlaces_totales_dir = grafodirigido.number_of_edges()
    num_enlaces_en_scc_ciclicas = 0
    for u, v in grafodirigido.edges():
        if u in nodos_en_scc_ciclicas and v in nodos_en_scc_ciclicas:
            num_enlaces_en_scc_ciclicas += 1

    cociente_enlaces_ciclicos = (num_enlaces_en_scc_ciclicas / num_enlaces_totales_dir) if num_enlaces_totales_dir > 0 else 0.0

    print("\n=== SCC Y ESTRUCTURA CÍCLICA (sobre la red dirigida original) ===")
    print(f"Número total de nodos dirigidos: {num_nodos_totales_dir}")
    print(f"Número de nodos en SCCs cíclicas: {num_nodos_ciclicos}")
    print(f"Cociente nodos cíclicos / nodos totales: {cociente_nodos_ciclicos:.4f}")
    print(f"Número total de enlaces dirigidos: {num_enlaces_totales_dir}")
    print(f"Número de enlaces con ambos extremos en SCCs cíclicas: {num_enlaces_en_scc_ciclicas}")
    print(f'Cociente enlaces "cíclicos" (en SCCs) / enlaces totales: {cociente_enlaces_ciclicos:.4f}\n')

    # ----------------------------------------------------------------
    # 1) UMBRAL y NODOS DESTACADOS ANTES DE REDUCIR LA RED
    #    (ANTES de eliminar ciclos y de hacer el DAG)
    # ----------------------------------------------------------------
    grados_original = dict(grafodirigido.degree())  # grado total (in-degree + out-degree)

    # Umbral que define qué saberes vas a pintar
    UMBRAL_GRADO = 0.75*nx.number_of_nodes(grafodirigido)  # ajustable

    nodos_destacados_original = [n for n, g in grados_original.items() if g > UMBRAL_GRADO]

    # print("\n=== NODOS DESTACADOS ANTES DE JERARQUIZAR (red completa, sin reducir) ===")
    # if not nodos_destacados_original:
    #     print(f"No hay nodos con grado > {UMBRAL_GRADO} en la red original.")
    # else:
    #     for nodo in sorted(nodos_destacados_original, key=lambda n: grados_original[n], reverse=True):
    #         print(f"{nodo}: grado original = {grados_original[nodo]}")

    # ----------------------------------------------------------------
    # 2) JERARQUIZAR (CONVERTIR A DAG) Y LUEGO VOLVER A TOMAR LOS MISMOS NODOS
    # ----------------------------------------------------------------

    # 2.1) Como ejemplo, calculamos un simple DAG eliminando aristas que generan ciclos
    # (en la red dirigida original grafodirigido)

    # grafoDAG = grafodirigido.copy()
    # while not nx.is_directed_acyclic_graph(grafoDAG):
    #     # tomamos un ciclo cualquiera
    #     ciclo_dirigido = nx.find_cycle(grafoDAG, orientation='original')
    #   #  print("Eliminando arista del ciclo:", ciclo_dirigido[0])
    #     u, v, *_ = ciclo_dirigido[0]
    #     # quitamos solo la primera arista del ciclo
    #     if grafoDAG.has_edge(u, v):
    #         grafoDAG.remove_edge(u, v)

    # graphparaelDAG = grafodirigidosimple.copy()
    # while not nx.is_directed_acyclic_graph(graphparaelDAG):
    #     # tomamos un ciclo cualquiera
    #     ciclo_dirigido = nx.find_cycle(graphparaelDAG, orientation='original')
    #     print("Eliminando arista del ciclo:", ciclo_dirigido[0])
    #     u, v = ciclo_dirigido[0][:2]  # ajustar si el ciclo tiene más componentes
    #     graphparaelDAG.remove_edge(u, v)



    ruta_excel = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\fisicaidentificador.xlsx"
    df = pd.read_excel(ruta_excel)
    diccionario_ids = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # nombre → ID


    # -------------------------
    # Eliminar ciclos (hacer DAG)
    # -------------------------
    ciclo = 0
    grafodirigido = grafodirigido.copy()
    while not nx.is_directed_acyclic_graph(grafodirigido):
        ciclo += 1

        try:
            cycle = nx.find_cycle(grafodirigido, orientation='original')
        except nx.NetworkXNoCycle:
            break

        if cycle:
            # quitamos solo la primera arista del ciclo
            u, v, *_ = cycle[0]
            if grafodirigido.has_edge(u, v):
                grafodirigido.remove_edge(u, v)
            else:
                print(f"La arista ({u}, {v}) no existe en el grafo.")


    # --------------------------------------------------------------
    # Comparar cantidad de caminos (enlaces) cíclicos vs acíclicos
    # --------------------------------------------------------------

    # En este punto:
    #  - 'ciclo' = número de enlaces eliminados para romper todos los ciclos
    #  - 'grafodirigido' ya es un DAG (solo tiene enlaces acíclicos, después de eliminar aristas de ciclos)

    num_enlaces_aciclicos = nx.number_of_edges(grafodirigido)
    num_enlaces_ciclicos = ciclo
    num_enlaces_originales = num_enlaces_aciclicos + num_enlaces_ciclicos

    print("\n=== COMPARACIÓN ENLACES CÍCLICOS vs ACÍCLICOS (RED DIRIGIDA) ===")
    print(f"Enlaces originales totales (antes de quitar ciclos): {num_enlaces_originales}")
    print(f"Enlaces que formaban parte de ciclos (eliminados): {num_enlaces_ciclicos}")
    print(f"Enlaces acíclicos (en el DAG final): {num_enlaces_aciclicos}")

    if num_enlaces_originales > 0:
        porc_ciclicos = 100 * num_enlaces_ciclicos / num_enlaces_originales
        porc_aciclicos = 100 * num_enlaces_aciclicos / num_enlaces_originales

        print(f"Porcentaje de enlaces cíclicos: {porc_ciclicos:.2f}%")
        print(f"Porcentaje de enlaces acíclicos: {porc_aciclicos:.2f}%")
    else:
        print("No hay enlaces en la red dirigida original.")


    # --------------------------------------------------------------
    # CURRÍCULUM: construir y dibujar grafo (cadenas por bloque) + comparar
    # --------------------------------------------------------------
    G_curriculum, curr_pos = construir_grafo_curriculum_cadenas_sin_interbloques(grafodirigido)

    # Dibujar el grafo del currículum (4 cadenas separadas)
    dibujar_grafo_curriculum(G_curriculum, curr_pos)

    # Comparación: tu DAG vs cronología del currículum (por bloque/orden)
    comparables, violaciones, ratio = comparar_dag_vs_curriculum(grafodirigido, curr_pos)


    # --------------------------------------------------------------
    # TAMAÑO DE LAS REDES: NODOS Y ARISTAS
    # --------------------------------------------------------------

    num_nodos_red = grafodirigido.number_of_nodes()
    num_aristas_red = grafodirigido.number_of_edges()

    num_nodos_curr = G_curriculum.number_of_nodes()
    num_aristas_curr = G_curriculum.number_of_edges()

    print("\n=== TAMAÑO DE LAS REDES ===")
    print(f"Tu red (DAG inferido):")
    print(f"  - Nodos: {num_nodos_red}")
    print(f"  - Aristas: {num_aristas_red}")

    print(f"\nRed del currículum:")
    print(f"  - Nodos: {num_nodos_curr}")
    print(f"  - Aristas: {num_aristas_curr}")

    print("\n=== COMPARACIÓN RED vs CURRÍCULUM ===")
    print(f"Nodos red: {len(curr_pos)}")
    print(f"Enlaces red jerárquica: {comparables}")

    if ratio is None:
        print("No hay aristas comparables (faltan nodos del currículum o no coinciden nombres).")
    else:
        print(f"Violaciones (En nuestra red u->v, pero el currículum no pone u antes que v): {len(violaciones)}")
        print(f"Porcentaje de violación: {100*ratio:.2f}%")

        def gravedad(vio):
            (_, _, (b_u, k_u), (b_v, k_v)) = vio
            return (b_u - b_v, k_u - k_v)

        top = sorted(violaciones, key=gravedad, reverse=True)[:15]
        print("\nTop 15 violaciones (más graves):")
        for u, v, pos_u, pos_v in top:
            print(f"- u: {u[:70]}...  ->  v: {v[:70]}... | curr(u)={pos_u}, curr(v)={pos_v}")



    # ----------------------------------------------------------------
    # 3) IMPRIMIR EL GRADO DE TODOS LOS NODOS (YA EN EL DAG)
    # ----------------------------------------------------------------
    grados_dag = dict(grafodirigido.degree())  # grado total (in-degree + out-degree)

    # print("\n=== GRADO DE TODOS LOS NODOS (DAG) ===")
    # for nodo, grado in sorted(grados_dag.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{nodo}: grado = {grado}")

    # ----------------------------------------------------------------
    # 4) Reetiquetar nodos con IDs del Excel
    # ----------------------------------------------------------------
    # En este punto, los nodos de grafodirigido siguen siendo los nombres originales.
    # Creamos la lista de IDs correspondiente a los nodos destacados en la red original.
    nodos_destacados_ids = []
    for nombre_original in nodos_destacados_original:
        if nombre_original in diccionario_ids:
            nodos_destacados_ids.append(diccionario_ids[nombre_original])
        else:
            print(f"Advertencia: El nodo '{nombre_original}' no tiene ID asociado en el Excel.")

    # Ahora creamos una nueva copia de la red total (reducida/DAG) con las etiquetas renombradas a IDs
    mapping = {}
    for nombre_original in grafodirigido.nodes():
        if nombre_original in diccionario_ids:
            mapping[nombre_original] = diccionario_ids[nombre_original]  # Nombre → ID
        else:
            mapping[nombre_original] = nombre_original  # Si no tiene ID, conservar el nombre

    grafodirigido_ids = nx.relabel_nodes(grafodirigido, mapping)

    # ----------------------------------------------------------------
    # 5) Visualizar o imprimir la jerarquía del DAG reetiquetado
    # ----------------------------------------------------------------
    if not nx.is_directed_acyclic_graph(grafodirigido_ids):
        print("\nAdvertencia: grafodirigido_ids no es un DAG. La jerarquía no será fiable.")
    else:
        jerarquia = jerarquia_nodos(grafodirigido_ids)
        # Ordenar nodos por jerarquía
        nodos_por_nivel = defaultdict(list)
        for nodo, nivel in jerarquia.items():
            nodos_por_nivel[nivel].append(nodo)

        print("\n=== Jerarquía de nodos (DAG con IDs) ===")
        for nivel in sorted(nodos_por_nivel.keys()):
            print(f"Nivel {nivel}: {nodos_por_nivel[nivel]}")

    # Sólo para debug: Dibujar un pequeño subgrafo si se desea
    # subgraph = get_analysis_subgraph(grafodirigido_ids, n=15)
    # pos = nx.spring_layout(subgraph, seed=42)
    # plt.figure(figsize=(8, 6))
    # nx.draw(subgraph, pos, with_labels=True, node_size=500, node_color='lightblue', arrows=True)
    # plt.title("Subgrafo de prueba (DAG con IDs)")
    # plt.show()


    # --------------------------------------------------------------------------------
    # 6) DIBUJAR LA JERARQUÍA CON LOS NODOS DESTACADOS (en rojo) SEGÚN LA RED ORIGINAL
    # --------------------------------------------------------------------------------
    # Sólo si grafodirigido_ids es un DAG
    if nx.is_directed_acyclic_graph(grafodirigido_ids):
        # Asignamos la jerarquía ya calculada como atributo de cada nodo
        for nodo, nivel in jerarquia.items():
            grafodirigido_ids.nodes[nodo]['nivel'] = nivel

        # Creamos una posición vertical por nivel (tipo 'capa')
        pos = {}
        niveles_unicos = sorted(set(jerarquia.values()))
        # Para cada nivel, distribuimos los nodos sobre el eje x
        for nivel in niveles_unicos:
            nodos_nivel = [n for n, lvl in jerarquia.items() if lvl == nivel]
            # Asignamos posiciones x equiespaciadas para los nodos de ese nivel
            x_coords = np.linspace(-1, 1, len(nodos_nivel))
            for x, nodo in zip(x_coords, nodos_nivel):
                pos[nodo] = (x, -nivel)  # y = -nivel para que se vea de arriba abajo

        # Construimos el conjunto de nodos destacados en el DAG reetiquetado
        nodos_destacados_ids_set = set(nodos_destacados_ids)

        # Preparamos colores y tamaños: nodos destacados (según la red original) en rojo y más grandes
        node_colors = []
        node_sizes = []
        for nodo in grafodirigido_ids.nodes():
            if nodo in nodos_destacados_ids_set:
                node_colors.append('red')
                node_sizes.append(500)
            else:
                node_colors.append('lightblue')
                node_sizes.append(300)

        plt.figure(figsize=(12, 8))
        # Dibujar nodos
        nx.draw_networkx_nodes(
            grafodirigido_ids,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )

        # Dibujar aristas
        nx.draw_networkx_edges(
            grafodirigido_ids,
            pos,
            arrowstyle='-|>',
            arrowsize=10,
            alpha=0.5
        )

        # Etiquetas de nodos
        nx.draw_networkx_labels(
            grafodirigido_ids,
            pos,
            font_size=8
        )

        # Si quieres, puedes colorear las aristas según algún criterio,
        # por ejemplo, diferenciando si van "hacia abajo" o "hacia arriba"
        # en términos de nivel, pero aquí se deja simple.
        #
        # Aquí podrías también graficar manualmente, con color distinto, los nodos
        # cuyas etiquetas provienen de nodos destacados en la red original.

    # # Crear una nueva figura
    # fig, ax = plt.subplots(figsize=(10, 8))  # Ajusta el tamaño según necesites
    #
    # # Dibujar los nodos
    # nx.draw(
    #     grafodirigido_ids, pos,
    #     with_labels=True,
    #     node_size=500,
    #     node_color=node_colors,
    #     edge_color='gray',
    #     width=1,
    #     font_size=8,
    #     ax=ax
    # )
    #
    # # Dibujar las etiquetas de las aristas
    # edge_labels = nx.get_edge_attributes(grafodirigido_ids, 'tipo')
    # nx.draw_networkx_edge_labels(grafodirigido_ids, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    #
    # # Limpiar los bordes blancos de las etiquetas
    # clean_edge_labels(ax)
    #
    # # Barra de color para la medida básico-aplicado
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    #
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('Medida "básico-aplicado" (out-degree - in-degree)', fontsize=12)
    #
    # plt.title("Jerarquía nodos red mates (nodos destacados según la red original en rojo)", fontsize=20, y=1.05)
    #
    # carpeta = r"C:\Users\pablo\Desktop\PAPERS\Imagenes"
    # nombre_archivo = "jerarquia_mates_red_total.png"
    # ruta_salida = os.path.join(carpeta, nombre_archivo)
    # #plt.savefig(ruta_salida, dpi=1200, bbox_inches='tight')
    #
    # plt.show()  # si quier es verlo en pantalla


    # ----------------------------------------------------------------
    # 7) Resumen final: ID → Nombre original (grado original)
    #    Solo para los nodos que superan el umbral en la red original
    # ----------------------------------------------------------------
    #print(f"\n=== NODOS CON GRADO ORIGINAL > {UMBRAL_GRADO}: ID → Nombre original (grado original) ===")
    #if not nodos_destacados_original:
    #    print("No hay nodos que cumplan la condición en la red original.")
   # else:
   #     for nombre_original in sorted(nodos_destacados_original, key=lambda n: grados_original[n], reverse=True):
    #        id_nodo = diccionario_ids.get(nombre_original, "SIN_ID")
     #       grado_orig = grados_original[nombre_original]
      #      print(f"{id_nodo} → {nombre_original} (grado original = {grado_orig})")



    print("The number of edges in DAG is:", nx.number_of_edges(grafodirigido))
    print("Número de enlaces eliminados para el DAG:", ciclo)

    ciclos = list(nx.simple_cycles(grafodirigido))
    num_ciclos = len(ciclos)
    print(f"Número de ciclos dirigidos en la red reducida: {num_ciclos}")

if __name__ == "__main__":
    main()
