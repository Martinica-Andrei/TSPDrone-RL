import numpy as np
import os
import torch
import random
from utils.options import ParseParams
from utils.env_no_comb import Env, DataGenerator
from model.nnets import Actor, Critic
from utils.agent import A2CAgent
import pandas as pd


def get_dist_matrix():
    batch_size = args["batch_size"]
    nr_nodes = args["n_nodes"]
    data_dir = args["data_dir"]
    path = f"{data_dir}/DroneTruck-size-{batch_size}-len-{nr_nodes}.txt"
    with open(path, 'r') as file:
        text = file.read()
    values = np.array(text.split())
    values = values.reshape(-1, 3)
    values = values[:, :2].astype(float)
    dist_matrix = np.array([[np.linalg.norm(a - b)
                           for b in values] for a in values])
    df = pd.DataFrame(dist_matrix, index=np.arange(
        1, len(dist_matrix) + 1), columns=np.arange(1, len(dist_matrix) + 1))
    return df


def get_total_distance(moves, dist_matrix):
    ans = 0
    for i in range(1, len(moves)):
        s, f = moves[i-1], moves[i]
        ans += dist_matrix.loc[s, f]
    return ans


#incarcarea fisierului se face prin args
#am modificat sa incarce DroneTruck-size-1-len-93.txt, el reprezinta coordonatele din fisierul orase
# 46,5670437 26,9145748 Bacau Judetul Bacau
# 2 46,2647012 26,782587 Onești judetul Bacau
# 3 46,4697857 26,4884629 Moinești judetul Bacau
# 4 46,4234767 26,4258031 Comănești judetul Bacau
# ...
# etc
# modelul asteapta ca fiecare rand sa fie o noua instanta/problema si fiecare 3 valori formeaza un nod (x, y, demand)
# "where x_i y_i d_i represents the x-y coordinate of customer i and demand. All demands are set to be 1.0 for customers. 
# The last components x_n y_n d_n represents the depot and d_n is set to 0.0 for the depot."
if __name__ == '__main__':
    args = ParseParams()
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d" % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    max_epochs = args['n_train']
    device = torch.device(
        "cuda") if torch.cuda.is_available else torch.device("cpu")
    save_path = args['save_path']
    n_nodes = args['n_nodes']
    dataGen = DataGenerator(args)
    data = dataGen.get_train_next()
    data = dataGen.get_test_all()
    env = Env(args, data)
    actor = Actor(args['hidden_dim'])
    critic = Critic(args['hidden_dim'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        path = save_path + 'n' + '100/best_model_actor_truck_params.pkl'
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location='cpu'))
            path = save_path + 'n' + '100/best_model_critic_params.pkl'
            critic.load_state_dict(torch.load(path, map_location='cpu'))
            print("Succesfully loaded keys")

    agent = A2CAgent(actor, critic, args, env, dataGen)
    R, truck_moves, drone_moves = agent.test()

    dist_matrix = get_dist_matrix()
    print("Distanta miscari camion folosind inteligenta artificiala cu drona: ")
    # truck_moves e 0 indexat prin urmare se adauga 1
    print(get_total_distance(truck_moves + 1, dist_matrix))
    print("Distanta miscari camion folosind gurobi fara drona: ")
    # truck_moves_without_drone e precalculat folosind gurobi
    truck_moves_without_drone = np.array([93, 81, 50, 69,  3,  6,  4, 11,  9, 19, 61, 35, 24,  8, 30, 15, 14,
                                          76, 48, 40, 16, 87, 65, 36,  7, 58, 18, 52, 22, 20,  2, 39, 84, 23,
                                          27, 90, 75, 64, 91, 60, 62, 72, 31, 56, 77, 49, 53, 51,  1, 46, 78,
                                          70, 88, 21, 85, 42, 63, 89, 32, 59, 82, 92, 34, 26, 86, 43, 29, 68,
                                          38, 54, 73, 25, 45, 67, 47, 57, 80, 74, 28, 55, 33, 13, 44, 41, 37,
                                          71,  5, 17, 83, 79, 10, 66, 12, 93])
    print(get_total_distance(truck_moves_without_drone, dist_matrix))

    print("In concluzie, pe setul de date orase, tsp doar cu camion parcurge o distanta mai mica decat inteligenta artificiala cu drona\n\n")

    print("Miscari camion: ")
    print(truck_moves + 1)
    print("Miscari drona: ")
    print(drone_moves)
