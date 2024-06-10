import random
import threading
import time
from ipywidgets import IntProgress
from IPython.display import display
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus  
import matplotlib.pyplot as plt
import numpy as np

class Generation_data:
    def __init__(self, nb_objets, nbres_camions):
        self.nb_objets = nb_objets
        self.nbres_camions = nbres_camions
        self.grand_tableau = self.generer_tableau()
        self.tableau_camion = self.gen_tab_camion()

    def generer_tableau(self):
        nb_villes = self.nb_objets * 2
        tab_ville = list(range(1, nb_villes + 1))
        grand_tableau = []

        for _ in range(nb_villes // 2):
            ville_collecte = tab_ville.pop(random.randint(0, len(tab_ville) - 1))
            ville_livraison = tab_ville.pop(random.randint(0, len(tab_ville) - 1))
            camion = random.randint(1, self.nbres_camions)
            grand_tableau.append([ville_collecte, ville_livraison, [camion]])

        return grand_tableau

    def gen_tab_camion(self):
        tab_camion = [[] for _ in range(self.nbres_camions)]

        for route in self.grand_tableau:
            camion_index = route[2][0] - 1
            tab_camion[camion_index].extend([route[0], route[1]])

        return tab_camion

    def affichage_camions(self):
        for i, camions in enumerate(self.tableau_camion):
            print(f"Camion {i + 1} : {camions}")

    def affichage_1_camion(self, id):
        return self.tableau_camion[id - 1]

def random_trajet_depart(villes_camion):
    # Construire une liste de couples (collecte, livraison)
    objets = [(villes_camion[i], villes_camion[i + 1]) for i in range(0, len(villes_camion), 2)]
    
    # Séparer les points de collecte et de livraison
    collectes = [collecte for collecte, _ in objets]
    livraisons = {livraison: collecte for collecte, livraison in objets}
    random.seed(a=3)

    # Mélanger aléatoirement les points de collecte
    random.shuffle(collectes)
    
    # Initialiser la liste d'itinéraire avec les collectes
    trajet = collectes[:]
    
    # Insérer chaque point de livraison à un index aléatoire après son point de collecte associé
    for livraison, collecte in livraisons.items():
        index_collecte = trajet.index(collecte)
        index_livraison = random.randint(index_collecte + 1, len(trajet))
        trajet.insert(index_livraison, livraison)
    
    return trajet
def random_temps_trajet(temps_min, temps_max, nb_villes):
    temps_trajet = []
    for i in range(nb_villes):
        ligne = []
        for j in range(nb_villes):
            if i == j:
                ligne.append(0)
            elif i < j:
                temps = random.randint(temps_min, temps_max)
                ligne.append(temps)
            else:
                ligne.append(temps_trajet[j][i])
        temps_trajet.append(ligne)
    return temps_trajet 

def solveur_simplex_camion(sommets, temps_trajet_routes):
    nb_villes = len(sommets)
    
    # Create the linear programming problem
    prob = LpProblem("TSP_Single_Truck", LpMinimize)
    
    # Decision variables: x[i,j] is 1 if the route from i to j is taken, otherwise 0
    x = LpVariable.dicts("x", (range(nb_villes), range(nb_villes)), cat='Binary')
    
    # Objective function: minimize the total travel time
    prob += lpSum(temps_trajet_routes[i][j] * x[i][j] for i in range(nb_villes) for j in range(nb_villes) if i != j)
    
    # Constraints: each city must be visited once for collection and once for delivery
    for i in range(nb_villes):
        prob += lpSum(x[i][j] for j in range(nb_villes) if i != j) == 1
        prob += lpSum(x[j][i] for j in range(nb_villes) if i != j) == 1
    
    # Solve the problem
    prob.solve()
    
    # Check for optimality and return the lower bound value
    if LpStatus[prob.status] == 'Optimal':
        return value(prob.objective)
    else:
        return None

def extract_submatrix(matrix, indices):
    submatrix = []
    for i in indices:
        row = []
        for j in indices:
            row.append(matrix[i-1][j-1])
        submatrix.append(row)
    return submatrix

def pheromones_update(pheromone_value, alpha, point_0):
    return pheromone_value * (1 - alpha) + alpha * point_0

def calculate_probabilities(current_index, points, pheromone, distance_matrix, alpha, beta, indextab):
    probabilities = []
    total = 0
    for i in range(len(points)):
        if i in indextab:
            probabilities.append(0)
        else:
            pheromone_level = pheromone[current_index][i] ** alpha
            visibility = (1 / distance_matrix[current_index][i]) ** beta if distance_matrix[current_index][i] > 0 else 0
            probability = pheromone_level * visibility
            total += probability
            probabilities.append(probability)
    
    if total == 0:
        return [1 / len(points)] * len(points)  # Si toutes les probabilités sont nulles, retourne des probabilités uniformes
    
    probabilities = [p / total for p in probabilities]
    return probabilities

def choose_next_index(probabilities):
    return np.random.choice(len(probabilities), p=probabilities)

def change_nodes(indextab, poidpath, distance_matrix, points, evaporation_rate, current_nodes, pheromone, alpha, beta):
    point_0 = current_nodes
    
    # Trouver l'index du point courant
    index = points.index(point_0)
    
    # Ajouter l'index courant à l'historique des index
    indextab.append(index)
    
    # Calculer les probabilités pour choisir le prochain index
    probabilities = calculate_probabilities(index, points, pheromone, distance_matrix, alpha, beta, indextab)
    
    # Choisir un nouvel index basé sur les probabilités
    newindex = choose_next_index(probabilities)
    
    # Mise à jour du poids du chemin
    if len(indextab) > 1:
        previousindex = indextab[-2]
    else:
        previousindex = index  # initialisez previousindex à index si c'est la première itération
    
    poidpath += distance_matrix[previousindex][index]
    #print(f'Poidpath après ajout: {poidpath}')
    
    # Mettre à jour l'index courant avec le nouvel index
    index = newindex
    point_0 = points[index]
    
    # Mettre à jour les phéromones
    pheromone[previousindex][index] = pheromones_update(pheromone[previousindex][index], alpha, 1.0)
    
    return point_0, poidpath

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate):
    best_path = []
    best_path_length = float('inf')
    ##distance_matrix = random_temps_trajet(40, 100, len(points))
    pheromone = initialize_pheromones(len(points))
    bar1 = IntProgress(min=0, max=n_iterations, layout={"width" : "100%"})
    display(bar1)
    for iteration in range(n_iterations):
        bar1.value += 1
        for ant in range(n_ants):
            path = []
            indextab = []
            poidpath = 0
            current_point = points[0]
            
            while len(path) < len(points):
                path.append(current_point)
                current_point, poidpath = change_nodes(indextab, poidpath, distance_matrix, points, evaporation_rate, current_point, pheromone, alpha, beta)
            path.append(points[0])
            poidpath += distance_matrix[points.index(current_point)][0]
            if poidpath < best_path_length:
                best_path_length = poidpath
                best_path = path
            
        pheromone = evaporate_pheromones(pheromone, evaporation_rate)
    print(f'Best path length: {best_path_length}')
    print(f'Best path: {best_path}')
    bar1.close()
    return best_path_length, best_path
def initialize_pheromones(num_nodes):  
    return np.ones((num_nodes, num_nodes))

def evaporate_pheromones(pheromone, evaporation_rate):
    return pheromone * (1 - evaporation_rate)



def mono_thread(k, n_ants, n_iterations, alpha, beta, evaporation_rate):
    camion_list = logistics.affichage_1_camion(k)
    print("camion ", k, " : ", camion_list)
    meilleur_temps,meilleur_itineraire = ant_colony_optimization(camion_list, n_ants, n_iterations, alpha, beta, evaporation_rate)
    with results_lock:
        results.append(meilleur_temps)
        r = [meilleur_itineraire, meilleur_temps]
        resultSimplexe.append(r)
        
def launch_threads(k, n_ants, n_iterations, alpha, beta, evaporation_rate):
    threads = []

    start_time = time.time()  # Start 

    for i in range(k):
        thread = threading.Thread(target=mono_thread, args=(i + 1, n_ants, n_iterations, alpha, beta, evaporation_rate))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    
    end_time = time.time()  # Stop
    
    total_duration = end_time - start_time
    print("")
    print(f"Longest duration for all threads: {total_duration:.4f} seconds")
    print("End of all the thread")
    index_max_time = max(range(len(resultSimplexe)), key=lambda i: resultSimplexe[i][1])
    camion_list = logistics.affichage_1_camion(index_max_time)
    temps_trajets_camions = extract_submatrix(distance_matrix, camion_list)
    borne_inf = solveur_simplex_camion(camion_list, temps_trajets_camions)
    rapprochement_borne = (borne_inf / max(results)) * 100
    resultSimplexe.clear()
    print("Results: ", results)
    return max(results), total_duration, rapprochement_borne

tab_result_glo = []
tab_result_glo_temp = []
tab_test = [5 ,10,  15, 20, 25, 30, 35, 45, 200,500]
rapprochements = []

for size in tab_test:
    nombres_objets = size
    nombres_camions = int(nombres_objets / 10)
    if nombres_camions<1:nombres_camions=1
    random.seed(a=3)
    distance_matrix = random_temps_trajet(40, 100, 2 * nombres_objets)
    n_ants = 20
    n_iterations = 200
    alpha = 1
    beta = 3
    evaporation_rate = 0.8
    nombres_villes = nombres_objets * 2
    logistics = Generation_data(nombres_objets, nombres_camions)
    results = []
    resultSimplexe = []
    results_lock = threading.Lock()
    var_sortie, var_temp, rapprochement = launch_threads(size, n_ants, n_iterations, alpha, beta, evaporation_rate)
    tab_result_glo.append(var_sortie)
    tab_result_glo_temp.append(var_temp)
    rapprochements.append(rapprochement)

plt.figure(figsize=(10, 6))
plt.plot(tab_test,rapprochements, marker='o', linestyle='-', color='b')
plt.title('Variation of Solution Quality as a Function of Instance Size')
plt.xlabel('Instance Size')
plt.ylabel('Solution Quality')
plt.grid(True)
plt.figure(figsize=(10, 6))
plt.plot(tab_test,tab_result_glo_temp, marker='o', linestyle='-', color='b')
plt.title('Variation of Execution Time as a Function of Instance Size')
plt.xlabel('Instance Size')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.show()