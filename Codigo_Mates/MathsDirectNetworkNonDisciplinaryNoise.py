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

# Solicitar la ruta del archivo CSV antes de ejecutar
archivo_csv = r"C:\Users\pablo\Desktop\PAPERS\Redes\redmates.csv"

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

    for fila in reader[1:]:
        if not fila:
            continue

        nodo_origen = fila[0]
        relaciones = fila[1:]

        relaciones = [int(valor) if valor.strip().isdigit() else 0 for valor in relaciones]

        for i, valor in enumerate(relaciones):
            nodo_destino = nombres_nodos[i + 1]

            # ⚠️ Evitar autoenlaces y valores 0
            if nodo_origen == nodo_destino or valor == 0:
                continue

            # 🔹 Agregar enlaces según las reglas
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


def clean_name(name):
    return name.split()[0]


def dibujar_grafo(G, Gsi, Gno, color, titulo):
    """Dibuja y guarda el grafo mixto con colores diferenciando aristas dirigidas y no dirigidas."""
    plt.figure(figsize=(19.2, 10.8))

    # 🔹 Disposición de nodos
    pos = nx.kamada_kawai_layout(G)

    # 🔹 Dibujar nodos
    sc = nx.draw_networkx_nodes(
        G, pos, node_size=1000, node_color=color,
        edgecolors="black", cmap=plt.cm.coolwarm
    )

    # 🔹 Aristas dirigidas
    edges_dirigidos = [(u, v) for u, v, d in Gsi.edges(data=True) if d.get("tipo") == "dirigido"]
    nx.draw_networkx_edges(
        Gsi, pos, edgelist=edges_dirigidos,
        edge_color="green", style="solid",
        arrows=True, arrowsize=15, width=2
    )

    # 🔹 Etiquetas
    nx.draw_networkx_labels(
        G, pos, font_size=6, font_color="black",
        font_weight="light", verticalalignment='center',
        horizontalalignment='center'
    )

    # 🔹 Barra de color
    cbar = plt.colorbar(sc, shrink=0.8)
    cbar.set_label(titulo, fontsize=12)

    nombre_archivo = f"{titulo.replace(' ', '_').lower()}.png"
    plt.title(titulo)
    plt.savefig(nombre_archivo, format="png", dpi=300)

    plt.show()

    print(f"\n✅ La imagen del grafo ha sido guardada como '{nombre_archivo}'.")


def main():
    grafodirigido, grafonodirigido, grafototal, grafototalsimple = cargar_grafo_desde_csv(archivo_csv)

    if grafodirigido is None:
        return

    print("Número de nodos en la red total:", nx.number_of_nodes(grafototal))
    print("Número de enlaces en la red total:", nx.number_of_edges(grafototal))

    # ----------------------------------------------------------------
    # 1) UMBRAL y NODOS DESTACADOS ANTES DE REDUCIR LA RED
    #    (ANTES de eliminar ciclos y de hacer el DAG)
    # ----------------------------------------------------------------
    grados_original = dict(grafototal.degree())  # grado total (in-degree + out-degree)

    # Umbral que define qué saberes vas a pintar
    UMBRAL_GRADO = 0.75*nx.number_of_nodes(grafototal)  # ajustable

    nodos_destacados_original = [n for n, g in grados_original.items() if g > UMBRAL_GRADO]

    print("\n=== NODOS DESTACADOS ANTES DE JERARQUIZAR (red completa, sin reducir) ===")
    if not nodos_destacados_original:
        print(f"No hay nodos con grado > {UMBRAL_GRADO} en la red original.")
    else:
        for nodo in sorted(nodos_destacados_original, key=lambda n: grados_original[n], reverse=True):
            print(f"{nodo}: grado original = {grados_original[nodo]}")

    # ----------------------------------------------------------------
    # 2) JERARQUÍA: Eliminar ciclos para obtener un DAG sobre la red dirigida
    # ----------------------------------------------------------------

    ruta_excel = r"C:\Users\pablo\Desktop\master\DIDÁCTICAS\TFM\matesidentificador.xlsx"
    df = pd.read_excel(ruta_excel)
    diccionario_ids = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # nombre → ID

    # -------------------------
    # Eliminar ciclos (hacer DAG)
    # -------------------------
    ciclo = 0
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

    print("Número de enlaces eliminados para hacer el DAG:", ciclo)

    # ----------------------------------------------------------------
    # 3) IMPRIMIR EL GRADO DE TODOS LOS NODOS (YA EN EL DAG)
    # ----------------------------------------------------------------
    grados_dag = dict(grafodirigido.degree())  # grado total (in-degree + out-degree)

    print("\n=== GRADO DE TODOS LOS NODOS (DAG) ===")
    for nodo, grado in sorted(grados_dag.items(), key=lambda x: x[1], reverse=True):
        print(f"{nodo}: grado = {grado}")

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
            print(f"Aviso: el nodo original '{nombre_original}' no se encontró en el Excel y no se podrá resaltar en la jerarquía.")

    # Ahora reetiquetamos la red con IDs
    grafodirigido = nx.relabel_nodes(grafodirigido, diccionario_ids)

    print("Nodos con nuevos IDs:", list(grafodirigido.nodes))
    print(f"Is this graph a DAG? {nx.is_directed_acyclic_graph(grafodirigido)}")
    topological_order = list(nx.topological_sort(grafodirigido))
    print(f"Topological Order: {topological_order}")

    # ----------------------------------------------------------------
    # 5) Medida básico-aplicado (sobre el DAG con IDs)
    # ----------------------------------------------------------------
    medida_basico_aplicado = {
        node: grafodirigido.out_degree(node) - grafodirigido.in_degree(node)
        for node in grafodirigido.nodes
    }

    import matplotlib.colors as mcolors

    valores = list(medida_basico_aplicado.values())
    vmin, vmax = min(valores), max(valores)

    # Por si acaso todos los nodos tienen el mismo valor
    if vmin == vmax:
        vmin -= 1
        vmax += 1

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.cividis  # azul-amarillo

    # Colores base según medida básico-aplicado
    colores_nodos = [cmap(norm(medida_basico_aplicado[n])) for n in grafodirigido.nodes]

    # ----------------------------------------------------------------
    # 6) Dibujo del grafo jerárquico
    #     - Todos los nodos coloreados por básico-aplicado
    #     - EN ROJO: los nodos que superaban el umbral en la red original
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(24, 14))

    # Disposición jerárquica usando Graphviz (dot)
    pos = graphviz_layout(grafodirigido, prog='dot')

    # 1) Dibujar todos los nodos con color según básico-aplicado
    nx.draw(
        grafodirigido, pos, with_labels=True, ax=ax,
        node_color=colores_nodos, edge_color="gray",
        node_size=2500, font_size=10, font_weight='bold', font_color='white',
    )

    # 2) Superponer en rojo los nodos que superaban el umbral en la red original (ya en versión ID)
    if nodos_destacados_ids:
        nx.draw_networkx_nodes(
            grafodirigido, pos, nodelist=nodos_destacados_ids, ax=ax,
            node_color="red", edgecolors="black", linewidths=2, node_size=2600
        )
    else:
        print(f"No hay nodos con grado mayor que {UMBRAL_GRADO} en la red original; no se resaltan nodos en rojo.")

    # Barra de color para la medida básico-aplicado
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Medida "básico-aplicado" (out-degree - in-degree)', fontsize=12)

    plt.title("Jerarquía nodos red mates (nodos destacados según la red original en rojo)", fontsize=20, y=1.05)

    carpeta = r"C:\Users\pablo\Desktop\PAPERS\Imagenes"
    nombre_archivo = "jerarquia_mates_red_dirigida.png"
    ruta_salida = os.path.join(carpeta, nombre_archivo)
    #plt.savefig(ruta_salida, dpi=1200, bbox_inches='tight')

    plt.show()  # si quieres verlo en pantalla

    print("The number of edges in DAG is:", nx.number_of_edges(grafodirigido))
    print("Número de enlaces eliminados para el DAG:", ciclo)

    # ----------------------------------------------------------------
    # 7) Resumen final: ID → Nombre original (grado original)
    #    Solo para los nodos que superan el umbral en la red original
    # ----------------------------------------------------------------
    print(f"\n=== NODOS CON GRADO ORIGINAL > {UMBRAL_GRADO}: ID → Nombre original (grado original) ===")
    if not nodos_destacados_original:
        print("No hay nodos que cumplan la condición en la red original.")
    else:
        for nombre_original in sorted(nodos_destacados_original, key=lambda n: grados_original[n], reverse=True):
            id_nodo = diccionario_ids.get(nombre_original, "SIN_ID")
            grado_orig = grados_original[nombre_original]
            print(f"{id_nodo} → {nombre_original} (grado original = {grado_orig})")


if __name__ == "__main__":
    main()
