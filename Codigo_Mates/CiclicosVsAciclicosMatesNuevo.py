import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from infomap import Infomap


# =========================
# CONFIGURACIÓN DE RUTAS
# =========================
archivo_csv = r"C:\Users\pablo\Desktop\PAPERS\Redes\redmates.csv"
ruta_excel = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\matesidentificador.xlsx"
carpeta_salida = r"C:\Users\pablo\Desktop\PAPERS\Imagenes"
nombre_archivo_salida = "jerarquia_matesprueba_red_total.png"

# =========================
# CONFIGURACIÓN DE COLORES Y GRUPOS DE NODOS
# =========================

COLORES_POR_GRUPO = {
    "non-disciplinary": "#C2F4FC",   #azul/actitudinal
}

NODOS_POR_GRUPO = {
    "non-disciplinary": {"MAT.1.3","MAT.1.4","MAT.2.2","MAT.2.7","MAT.3.5","MAT.3.6","MAT.6.1","MAT.6.3","MAT.1.5","MAT.2.8","MAT.3.8","MAT.3.16","MAT.4.1","MAT.5.6","MAT.6.4","MAT.3.11","MAT.2.6","MAT.3.7","MAT.3.15","MAT.4.6","MAT.5.5"},
}

COLOR_RESTO_NODOS = "#F09886" #verde/disciplinar

TAM_NODOS_GRUPOS = 900
TAM_RESTO_NODOS = 900


# =========================
# FUNCIONES AUXILIARES
# =========================
def cargar_grafo_desde_csv(ruta_csv):
    """
    Carga un grafo mixto desde un CSV con nodos dirigidos y no dirigidos.

    Convención de valores:
        0 -> sin relación
        1 -> fila -> columna
        2 -> columna -> fila
        3 -> relación no dirigida
    """
    if not os.path.exists(ruta_csv):
        print(f"Error: El archivo {ruta_csv} no existe.")
        return None, None, None, None

    with open(ruta_csv, "r", newline="", encoding="utf-8") as file:
        reader = list(csv.reader(file))

    reader = [fila for fila in reader if any(fila)]
    if not reader:
        print("Error: el CSV está vacío.")
        return None, None, None, None

    nombres_nodos = reader[0]

    G = nx.DiGraph()
    Gno = nx.Graph()
    Gtotal = nx.DiGraph()
    Gtotalsimple = nx.Graph()

    for nombre in nombres_nodos[1:]:
        G.add_node(nombre)
        Gno.add_node(nombre)
        Gtotal.add_node(nombre)
        Gtotalsimple.add_node(nombre)

    for i, fila in enumerate(reader[1:], start=1):
        if i >= len(nombres_nodos):
            continue

        nodo_origen = nombres_nodos[i]

        for j, valor in enumerate(fila[1:], start=1):
            if j >= len(nombres_nodos):
                continue

            nodo_destino = nombres_nodos[j]

            if nodo_origen == nodo_destino or valor == "":
                continue

            try:
                valor = int(valor)
            except ValueError:
                continue

            if valor == 0:
                continue
            elif valor == 1:
                G.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")
            elif valor == 2:
                G.add_edge(nodo_destino, nodo_origen, tipo="dirigido")
                Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_destino, nodo_origen, tipo="simple")
            elif valor == 3:
                Gno.add_edge(nodo_origen, nodo_destino, tipo="no dirigido")
                Gtotal.add_edge(nodo_origen, nodo_destino, tipo="dirigido")
                Gtotal.add_edge(nodo_destino, nodo_origen, tipo="dirigido")
                Gtotalsimple.add_edge(nodo_origen, nodo_destino, tipo="simple")

    return G, Gno, Gtotal, Gtotalsimple


def clean_edge_labels(ax):
    """
    Elimina los cuadros blancos detrás de las etiquetas de aristas.
    """
    for artist in ax.get_children():
        if isinstance(artist, plt.Text):
            artist.set_bbox(dict(facecolor="none", edgecolor="none"))


def filtrar_red_por_grado(grafo, umbral_grado):
    """
    Devuelve un subgrafo con los nodos cuyo grado es mayor al umbral.
    """
    grados = dict(grafo.degree())
    nodos_filtrados = [n for n, g in grados.items() if g > umbral_grado]
    return grafo.subgraph(nodos_filtrados).copy()


def jerarquia_nodos(grafo):
    """
    Devuelve un diccionario con la jerarquía de los nodos en el DAG,
    basada en la longitud máxima de los caminos hacia cada nodo.
    """
    if not nx.is_directed_acyclic_graph(grafo):
        raise ValueError("El grafo debe ser un DAG para calcular la jerarquía.")

    jerarquia = {}
    for nodo in nx.topological_sort(grafo):
        predecesores = list(grafo.predecessors(nodo))
        if not predecesores:
            jerarquia[nodo] = 0
        else:
            jerarquia[nodo] = 1 + max(jerarquia[p] for p in predecesores)

    return jerarquia


def get_analysis_subgraph(G_original, n=10):
    """
    Obtiene un subgrafo a partir de los n nodos con mayor grado.
    """
    degree_dict = dict(G_original.degree())
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:n]
    subgraph = G_original.subgraph(top_nodes).copy()
    return subgraph


def calcular_metricas_e_infomap(G):
    """
    Calcula métricas de centralidad sobre el grafo G, exporta a Pajek
    y detecta comunidades con Infomap.
    """
    if not isinstance(G, (nx.DiGraph, nx.Graph)):
        raise TypeError("G debe ser un grafo de NetworkX.")

    grado_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    if isinstance(G, nx.DiGraph):
        pagerank_centrality = nx.pagerank(G)
    else:
        pagerank_centrality = None

    clustering_centrality = nx.clustering(G.to_undirected() if isinstance(G, nx.DiGraph) else G)

    if not isinstance(G, nx.DiGraph):
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    else:
        eigenvector_centrality = None

    filename_base = "red_network"
    pajek_path = filename_base + ".net"
    nx.write_pajek(G, pajek_path)
    print(f"Grafo exportado a {pajek_path} para Infomap.")

    data = {
        "node": list(G.nodes()),
        "degree": [G.degree(n) for n in G.nodes()],
        "degree_centrality": [grado_centrality[n] for n in G.nodes()],
        "betweenness": [betweenness_centrality[n] for n in G.nodes()],
        "closeness": [closeness_centrality[n] for n in G.nodes()],
        "clustering": [clustering_centrality[n] for n in G.nodes()],
    }

    if pagerank_centrality is not None:
        data["pagerank"] = [pagerank_centrality[n] for n in G.nodes()]

    if eigenvector_centrality is not None:
        data["eigenvector"] = [eigenvector_centrality[n] for n in G.nodes()]

    df = pd.DataFrame(data)
    df_path = filename_base + "_metrics.csv"
    df.to_csv(df_path, index=False)
    print(f"Métricas de centralidad guardadas en {df_path}.")

    im = Infomap()
    for u, v in G.edges():
        im.add_link(str(u), str(v))
    im.run()

    communities = defaultdict(list)
    for node in im.nodes:
        communities[node.module_id].append(node.node_id)

    communities_path = filename_base + "_communities.txt"
    with open(communities_path, "w", encoding="utf-8") as f:
        for module_id, nodes in communities.items():
            f.write(f"Community {module_id}: {', '.join(map(str, nodes))}\n")
    print(f"Comunidades detectadas con Infomap guardadas en {communities_path}.")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="blue", alpha=0.7)
    nx.draw_networkx_edges(G, pos, arrows=isinstance(G, nx.DiGraph), alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Visualización del grafo")
    plt.axis("off")
    plt.show()


def get_top_nodes_by_centrality(G, metric="degree", top_n=10):
    """
    Obtiene los top_n nodos por la métrica de centralidad especificada.
    """
    if metric == "degree":
        centrality = dict(G.degree())
    elif metric == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif metric == "closeness":
        centrality = nx.closeness_centrality(G)
    elif metric == "pagerank":
        if isinstance(G, nx.DiGraph):
            centrality = nx.pagerank(G)
        else:
            raise ValueError("PageRank solo está definido aquí para grafos dirigidos.")
    elif metric == "eigenvector":
        if not isinstance(G, nx.DiGraph):
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        else:
            raise ValueError("Eigenvector centrality se implementa aquí solo para no dirigidos.")
    else:
        raise ValueError(f"Métrica '{metric}' no reconocida.")

    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_nodes


# =========================
# MAIN
# =========================
def main():
    grafodirigido, grafonodirigido, grafototal, grafototalsimple = cargar_grafo_desde_csv(archivo_csv)

    if grafodirigido is None:
        return

    print("Número de nodos en la red dirigida:", nx.number_of_nodes(grafodirigido))
    print("Número de enlaces en la red dirigida:", nx.number_of_edges(grafodirigido))

    # ---------------------------------------------------------
    # SCC Y ESTRUCTURA CÍCLICA
    # ---------------------------------------------------------
    sccs = list(nx.strongly_connected_components(grafodirigido))

    nodos_en_scc_ciclicas = set()
    for comp in sccs:
        if len(comp) > 1:
            nodos_en_scc_ciclicas.update(comp)
        elif len(comp) == 1:
            nodo = next(iter(comp))
            if grafodirigido.has_edge(nodo, nodo):
                nodos_en_scc_ciclicas.add(nodo)

    num_nodos_totales_dir = grafodirigido.number_of_nodes()
    num_nodos_ciclicos = len(nodos_en_scc_ciclicas)
    cociente_nodos_ciclicos = (
        num_nodos_ciclicos / num_nodos_totales_dir if num_nodos_totales_dir > 0 else 0.0
    )

    num_enlaces_totales_dir = grafodirigido.number_of_edges()
    num_enlaces_en_scc_ciclicas = 0
    for u, v in grafodirigido.edges():
        if u in nodos_en_scc_ciclicas and v in nodos_en_scc_ciclicas:
            num_enlaces_en_scc_ciclicas += 1

    cociente_enlaces_ciclicos = (
        num_enlaces_en_scc_ciclicas / num_enlaces_totales_dir if num_enlaces_totales_dir > 0 else 0.0
    )

    print("\n=== SCC Y ESTRUCTURA CÍCLICA (red dirigida original) ===")
    print(f"Número total de nodos dirigidos: {num_nodos_totales_dir}")
    print(f"Número de nodos en SCCs cíclicas: {num_nodos_ciclicos}")
    print(f"Cociente nodos cíclicos / nodos totales: {cociente_nodos_ciclicos:.4f}")
    print(f"Número total de enlaces dirigidos: {num_enlaces_totales_dir}")
    print(f"Número de enlaces con ambos extremos en SCCs cíclicas: {num_enlaces_en_scc_ciclicas}")
    print(f'Cociente enlaces "cíclicos" / enlaces totales: {cociente_enlaces_ciclicos:.4f}\n')

    # ---------------------------------------------------------
    # UMBRAL Y NODOS DESTACADOS
    # ---------------------------------------------------------
    grados_original = dict(grafodirigido.degree())
    UMBRAL_GRADO = 0.75 * nx.number_of_nodes(grafodirigido)
    nodos_destacados_original = [n for n, g in grados_original.items() if g > UMBRAL_GRADO]

    # ---------------------------------------------------------
    # CARGAR EXCEL DE IDENTIFICADORES
    # ---------------------------------------------------------
    if not os.path.exists(ruta_excel):
        print(f"Error: el archivo Excel {ruta_excel} no existe.")
        return

    df = pd.read_excel(ruta_excel)
    diccionario_ids = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    # ---------------------------------------------------------
    # ELIMINAR CICLOS PARA CREAR DAG
    # ---------------------------------------------------------
    ciclo = 0
    grafodirigido_dag = grafodirigido.copy()

    while not nx.is_directed_acyclic_graph(grafodirigido_dag):
        try:
            cycle = nx.find_cycle(grafodirigido_dag, orientation="original")
        except nx.NetworkXNoCycle:
            break

        if cycle:
            u, v, *_ = cycle[0]
            if grafodirigido_dag.has_edge(u, v):
                grafodirigido_dag.remove_edge(u, v)
                ciclo += 1

    # ---------------------------------------------------------
    # COMPARAR ENLACES CÍCLICOS VS ACÍCLICOS
    # ---------------------------------------------------------
    num_enlaces_aciclicos = nx.number_of_edges(grafodirigido_dag)
    num_enlaces_ciclicos = ciclo
    num_enlaces_originales = num_enlaces_aciclicos + num_enlaces_ciclicos

    print("\n=== COMPARACIÓN ENLACES CÍCLICOS vs ACÍCLICOS ===")
    print(f"Enlaces originales totales: {num_enlaces_originales}")
    print(f"Enlaces que formaban parte de ciclos (eliminados): {num_enlaces_ciclicos}")
    print(f"Enlaces acíclicos (DAG final): {num_enlaces_aciclicos}")

    if num_enlaces_originales > 0:
        porc_ciclicos = 100 * num_enlaces_ciclicos / num_enlaces_originales
        porc_aciclicos = 100 * num_enlaces_aciclicos / num_enlaces_originales
        print(f"Porcentaje de enlaces cíclicos: {porc_ciclicos:.2f}%")
        print(f"Porcentaje de enlaces acíclicos: {porc_aciclicos:.2f}%")
    else:
        print("No hay enlaces en la red dirigida original.")

    # ---------------------------------------------------------
    # REETIQUETAR NODOS CON IDs
    # ---------------------------------------------------------
    nodos_destacados_ids = []
    for nombre_original in nodos_destacados_original:
        if nombre_original in diccionario_ids:
            nodos_destacados_ids.append(diccionario_ids[nombre_original])
        else:
            print(f"Advertencia: el nodo '{nombre_original}' no tiene ID asociado en el Excel.")

    mapping = {}
    for nombre_original in grafodirigido_dag.nodes():
        if nombre_original in diccionario_ids:
            mapping[nombre_original] = diccionario_ids[nombre_original]
        else:
            mapping[nombre_original] = nombre_original

    grafodirigido_ids = nx.relabel_nodes(grafodirigido_dag, mapping)


    # ---------------------------------------------------------
    # JERARQUÍA
    # ---------------------------------------------------------
    if not nx.is_directed_acyclic_graph(grafodirigido_ids):
        print("\nAdvertencia: grafodirigido_ids no es un DAG.")
        return

    jerarquia = jerarquia_nodos(grafodirigido_ids)
    nodos_por_nivel = defaultdict(list)
    for nodo, nivel in jerarquia.items():
        nodos_por_nivel[nivel].append(nodo)

    print("\n=== Jerarquía de nodos (DAG con IDs) ===")
    for nivel in sorted(nodos_por_nivel.keys()):
        print(f"Nivel {nivel}: {nodos_por_nivel[nivel]}")

    for nodo, nivel in jerarquia.items():
        grafodirigido_ids.nodes[nodo]["nivel"] = nivel

    # ---------------------------------------------------------
    # POSICIONES POR NIVEL
    # ---------------------------------------------------------

    SEPARACION_VERTICAL = 5  # distancia entre niveles
    ANCHO_NIVEL = 6  # separación horizontal entre nodos del mismo nivel

    pos = {}
    niveles_unicos = sorted(set(jerarquia.values()))

    for nivel in niveles_unicos:
        nodos_nivel = [n for n, lvl in jerarquia.items() if lvl == nivel]

        x_coords = np.linspace(-ANCHO_NIVEL, ANCHO_NIVEL, len(nodos_nivel)) if len(nodos_nivel) > 1 else [0]

        for x, nodo in zip(x_coords, nodos_nivel):
            pos[nodo] = (x, -nivel * SEPARACION_VERTICAL)

    # ---------------------------------------------------------
    # COLORES FIJOS SEGÚN EL GRUPO DEL NODO
    # ---------------------------------------------------------
    node_colors = []
    node_sizes = []

    for nodo in grafodirigido_ids.nodes():
        color_asignado = COLOR_RESTO_NODOS
        tam_asignado = TAM_RESTO_NODOS

        for grupo, nodos_grupo in NODOS_POR_GRUPO.items():
            if nodo in nodos_grupo:
                color_asignado = COLORES_POR_GRUPO[grupo]
                tam_asignado = TAM_NODOS_GRUPOS
                break

        node_colors.append(color_asignado)
        node_sizes.append(tam_asignado)

    # ---------------------------------------------------------
    # DIBUJO
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 12))

    nx.draw_networkx_nodes(
        grafodirigido_ids,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        ax=ax
    )

    nx.draw_networkx_edges(
        grafodirigido_ids,
        pos,
        edge_color="gray",
        width=1,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        alpha=0.6,
        ax=ax
    )

    nx.draw_networkx_labels(
        grafodirigido_ids,
        pos,
        font_size=8,
        ax=ax
    )

    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(color="#C2F4FC", label="non-disciplinary"),
        mpatches.Patch(color="#F09886", label="disciplinary")
    ]

    ax.legend(handles=legend_elements, title="BK type", bbox_to_anchor=(0.8, 1.12), loc="upper left")

    # edge_labels = nx.get_edge_attributes(grafodirigido_ids, "tipo")
    # nx.draw_networkx_edge_labels(
    #     grafodirigido_ids,
    #     pos,
    #     edge_labels=edge_labels,
    #     font_size=7,
    #     ax=ax
    # )

    clean_edge_labels(ax)

    plt.title(
        "Hierarchy of nodes in the mathematics network\n"
        "Custom color for selected nodes",
        fontsize=16,
        y=1.03
    )
    plt.axis("off")

    # ---------------------------------------------------------
    # GUARDAR FIGURA
    # ---------------------------------------------------------
    os.makedirs(carpeta_salida, exist_ok=True)
    ruta_salida = os.path.join(carpeta_salida, nombre_archivo_salida)
    plt.savefig(ruta_salida, dpi=600, bbox_inches="tight")
    print(f"\nImagen guardada en: {ruta_salida}")

    plt.show()

    # ---------------------------------------------------------
    # RESUMEN FINAL
    # ---------------------------------------------------------
    print("\n=== NODOS CON GRADO ORIGINAL SUPERIOR AL UMBRAL ===")
    if not nodos_destacados_original:
        print("No hay nodos que cumplan la condición en la red original.")
    else:
        for nombre_original in sorted(nodos_destacados_original, key=lambda n: grados_original[n], reverse=True):
            id_nodo = diccionario_ids.get(nombre_original, "SIN_ID")
            grado_orig = grados_original[nombre_original]
            print(f"{id_nodo} -> {nombre_original} (grado original = {grado_orig})")

    print("\nThe number of edges in DAG is:", nx.number_of_edges(grafodirigido_dag))
    print("Número de enlaces eliminados para el DAG:", ciclo)

    ciclos = list(nx.simple_cycles(grafodirigido_dag))
    num_ciclos = len(ciclos)
    print(f"Número de ciclos dirigidos en la red reducida: {num_ciclos}")


if __name__ == "__main__":
    main()