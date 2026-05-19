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

def louvain_partition_to_communities(partition_dict):
    """
    Convierte un dict {nodo: id_comunidad} en una lista de sets [{...}, {...}, ...]
    compatible con NetworkX (coverage/performance).
    """
    comms = defaultdict(set)
    for node, cid in partition_dict.items():
        comms[cid].add(node)
    return list(comms.values())


def coverage_manual(G, communities):
    """
    Coverage = fracción de aristas que caen dentro de comunidades
    """
    intra_edges = 0
    for community in communities:
        intra_edges += G.subgraph(community).number_of_edges()

    total_edges = G.number_of_edges()
    return intra_edges / total_edges if total_edges > 0 else 0


def performance_manual(G, communities):
    """
    Performance según definición clásica (Newman)
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = i

    intra_edges = 0
    inter_non_edges = 0

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            same_comm = node_to_comm[u] == node_to_comm[v]
            has_edge = G.has_edge(u, v)

            if same_comm and has_edge:
                intra_edges += 1
            elif not same_comm and not has_edge:
                inter_non_edges += 1

    total_pairs = n * (n - 1) / 2
    return (intra_edges + inter_non_edges) / total_pairs

def analizar_comunidades_louvain(G_undirected, n_runs=20, seed=42):
    """
    Ejecuta Louvain sobre un grafo NO dirigido de forma más estable:
    - Ejecuta n_runs veces (multi-arranque) con semillas distintas (derivadas de seed)
    - Elige la partición con mayor modularidad
    Devuelve:
    - partition dict nodo->comunidad
    - communities list[set]
    - coverage (manual)
    - performance (manual)
    - modularity (extra, por si la quieres reportar)
    """
    if G_undirected is None or G_undirected.number_of_nodes() == 0:
        raise ValueError("El grafo está vacío o es None.")
    if G_undirected.number_of_edges() == 0:
        raise ValueError("El grafo no tiene aristas (no tiene sentido Louvain).")

    best_partition = None
    best_Q = -1e9

    # Semillas reproducibles
    rng = np.random.default_rng(seed)
    seeds = rng.integers(low=0, high=2**31 - 1, size=n_runs, dtype=np.int64)

    for s in seeds:
        # Intento 1: si la versión soporta random_state
        try:
            part = community_louvain.best_partition(G_undirected, random_state=int(s))
        except TypeError:
            # Fallback: si NO soporta random_state, al menos fijamos RNG global
            random.seed(int(s))
            np.random.seed(int(s))
            part = community_louvain.best_partition(G_undirected)

        Q = community_louvain.modularity(part, G_undirected)

        if Q > best_Q:
            best_Q = Q
            best_partition = part

    communities = louvain_partition_to_communities(best_partition)
    cov = coverage_manual(G_undirected, communities)
    perf = performance_manual(G_undirected, communities)

    return best_partition, communities, cov, perf, best_Q

def eliminar_nodos_transversales(G, nodos_a_eliminar):
    H = G.copy()
    presentes = [n for n in nodos_a_eliminar if n in H]
    ausentes = [n for n in nodos_a_eliminar if n not in H]

    H.remove_nodes_from(presentes)

    return H, presentes, ausentes

def permutation_test_direct(G, nodos_transversales, n_iter=500, seed=42, two_sided=True):
    """
    Compara coverage/performance de G_trans (eliminación específica) con
    la distribución de coverage/performance al eliminar k nodos aleatorios.

    Devuelve p-valores y distribuciones nulas de cov/perf (NO de deltas).
    """
    random.seed(seed)
    np.random.seed(seed)

    # usar solo los nodos realmente presentes
    nodos_transversales = [n for n in nodos_transversales if n in G]
    k = len(nodos_transversales)
    if k == 0:
        raise ValueError("No hay nodos transversales presentes en el grafo.")
    if k >= G.number_of_nodes():
        raise ValueError("k es demasiado grande para el número de nodos del grafo.")

    # Observado: red sin transversales
    G_obs = G.copy()
    G_obs.remove_nodes_from(nodos_transversales)
    _, _, cov_obs, perf_obs, _ = analizar_comunidades_louvain(G_obs)

    # Nulo: eliminar k nodos al azar
    nodes = list(G.nodes())
    cov_rand = []
    perf_rand = []

    for _ in range(n_iter):
        rnd = random.sample(nodes, k)
        G_r = G.copy()
        G_r.remove_nodes_from(rnd)

        if G_r.number_of_nodes() < 2 or G_r.number_of_edges() == 0:
            continue

        _, _, cov_r, perf_r, _ = analizar_comunidades_louvain(G_r)
        cov_rand.append(cov_r)
        perf_rand.append(perf_r)

    cov_rand = np.array(cov_rand, dtype=float)
    perf_rand = np.array(perf_rand, dtype=float)

    if len(cov_rand) == 0:
        raise ValueError("No hay iteraciones válidas en el test aleatorio (grafo demasiado pequeño tras eliminar).")

    # p-valores empíricos
    if two_sided:
        # bilateral alrededor de la media nula (alternativa robusta y común en randomización)
        mu_cov = cov_rand.mean()
        mu_perf = perf_rand.mean()
        p_cov = (np.sum(np.abs(cov_rand - mu_cov) >= abs(cov_obs - mu_cov)) + 1) / (len(cov_rand) + 1)
        p_perf = (np.sum(np.abs(perf_rand - mu_perf) >= abs(perf_obs - mu_perf)) + 1) / (len(perf_rand) + 1)
    else:
        # unilateral "más extremo hacia arriba" (si tu hipótesis es que sube)
        p_cov = (np.sum(cov_rand >= cov_obs) + 1) / (len(cov_rand) + 1)
        p_perf = (np.sum(perf_rand >= perf_obs) + 1) / (len(perf_rand) + 1)

    return {
        "k": k,
        "n_iter_valid": len(cov_rand),
        "cov_obs": cov_obs,
        "perf_obs": perf_obs,
        "cov_rand": cov_rand,
        "perf_rand": perf_rand,
        "p_cov": p_cov,
        "p_perf": p_perf,
    }

def main():
    grafodirigido, grafonodirigido, grafototal, grafototalsimple = cargar_grafo_desde_csv(archivo_csv)

    if grafototal is None:
        return

    G = grafototalsimple  # Louvain sobre no dirigido
    NODOS_TRANSVERSALES = [
        "La conservación de la energía mecánica.",
        "Identificación en la naturaleza y aplicaciones.",
        "Contaminación acústica y otras aplicaciones.",
        "Situaciones y contextos naturales en los cuales se ponen de manifiesto diferentes fenómenos ondulatorios. Interferencias y difracción. Aplicaciones. Cambios en las propiedades de las ondas en función del desplazamiento del emisor y receptor.",
        "Esquema del espectro electromagnético, presencia en el entorno tecnológico y escala comparativa.",
        "Aplicaciones de la óptica geométrica",
        "Papel de la física cuántica en aplicaciones como el láser, resonancias magnéticas o nanotecnología.",
        "Otras aplicaciones en los campos de la ingeniería, la tecnología y la salud.",
        "Implicaciones en el cambio de paradigma de la mecánica clásica.",
        "Controversias históricas originadas por la naturaleza de la materia y la energía, derivadas de la dualidad onda-corpúsculo en la luz.",
    ]



    # --- RED ORIGINAL ---
    partition, communities, cov, perf, Q = analizar_comunidades_louvain(G, n_runs=30, seed=42)
    print("Modularidad (Q):", Q)
    print("\n=== ORIGINAL ===")
    print("Nodos:", G.number_of_nodes(), "Enlaces:", G.number_of_edges())
    print("Número de comunidades:", len(communities))
    print("Coverage:", cov)
    print("Performance:", perf)
    print("Tamaños:", sorted([len(c) for c in communities], reverse=True))

    # --- ELIMINAR NODOS TRANSVERSALES ---
    G2, eliminados, no_encontrados = eliminar_nodos_transversales(G, NODOS_TRANSVERSALES)

    print("\n=== ELIMINACIÓN ===")
    print("Nodos listados para eliminar:", len(NODOS_TRANSVERSALES))
    print("Nodos eliminados (presentes):", len(eliminados))
    if no_encontrados:
        print("Ojo: estos nodos no estaban en la red:", no_encontrados)

    # (opcional) componentes conexas
    try:
        print("Componentes conexas:", nx.number_connected_components(G2))
    except Exception:
        pass

    # --- DESPUÉS ---
    partition2, communities2, cov2, perf2, Q2 = analizar_comunidades_louvain(G2, n_runs=30, seed=42)
    print("Modularidad (Q):", Q2)
    print("\n=== SIN NODOS TRANSVERSALES ===")
    print("Nodos:", G2.number_of_nodes(), "Enlaces:", G2.number_of_edges())
    print("Número de comunidades:", len(communities2))
    print("Coverage:", cov2)
    print("Performance:", perf2)
    print("Tamaños:", sorted([len(c) for c in communities2], reverse=True))

    # --- PERMUTATION TEST ---
    print("\n=== TEST DIRECTO: G_trans vs G_random ===")
    res2 = permutation_test_direct(G, eliminados, n_iter=1000, seed=42, two_sided=True)

    print("k:", res2["k"])
    print("iteraciones válidas:", res2["n_iter_valid"])
    print("Coverage observado (sin transversales):", res2["cov_obs"])
    print("Performance observado (sin transversales):", res2["perf_obs"])
    print("p(Coverage):", res2["p_cov"])
    print("p(Performance):", res2["p_perf"])

    print("Coverage nulo: media=", float(res2["cov_rand"].mean()),
          "IC95%=", (float(np.percentile(res2["cov_rand"], 2.5)), float(np.percentile(res2["cov_rand"], 97.5))))
    print("Performance nulo: media=", float(res2["perf_rand"].mean()),
          "IC95%=", (float(np.percentile(res2["perf_rand"], 2.5)), float(np.percentile(res2["perf_rand"], 97.5))))


if __name__ == "__main__":
    main()
