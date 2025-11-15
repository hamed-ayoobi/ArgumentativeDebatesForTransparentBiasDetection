import copy

import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
from utility import fprint, methods
from enum import Enum

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import re

def extract_arg_name(arg_str):
    """Extracts the argument name from a string like 'Argument(name=Argument2, weight=0.4, strength=0.4)'."""
    match = re.search(r'name=([^,]+)', arg_str)
    return match.group(1) if match else arg_str

def compute_levels(G):
    """Assigns a level to each node: leaves get level 0, their parents get 1, etc."""
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    level = {n: 0 for n in leaves}
    queue = deque(leaves)
    while queue:
        node = queue.popleft()
        for pred in G.predecessors(node):
            if pred not in level or level[pred] < level[node] + 1:
                level[pred] = level[node] + 1
                queue.append(pred)
    return level

def plot_hierarchical_arguments(model, remove_unimportant_nodes=True):
    arguments = model.arguments
    G = nx.DiGraph()
    node_labels = {}
    node_strengths = {}


    # Add all nodes
    for arg in arguments:
        if arg.strength > 0 if remove_unimportant_nodes else True:
            G.add_node(arg.name)
            node_labels[arg.name] = f"{arg.name}\nStrength: {arg.strength:.2f}"
            node_strengths[arg.name] = arg.strength

    # Add edges for attackers (red dashed) and supporters (green solid)
    edge_styles = {}
    for arg in arguments:
        if arg.strength > 0 if remove_unimportant_nodes else True:
            for attacker, weight in arg.attackers.items():
                if attacker.strength > 0 if remove_unimportant_nodes else True:
                    attacker_name = attacker.name
                    G.add_edge(attacker_name, arg.name)
                    edge_styles[(attacker_name, arg.name)] = {
                        'color': 'red', 'style': 'dashed', 'weight': weight * attacker.strength
                    }
            for supporter, weight in arg.supporters.items():
                if supporter.strength > 0 if remove_unimportant_nodes else True:
                    supporter_name = supporter.name
                    G.add_edge(supporter_name, arg.name)
                    edge_styles[(supporter_name, arg.name)] = {
                        'color': 'green', 'style': 'solid', 'weight': weight * supporter.strength
                    }

    if remove_unimportant_nodes:
        # --- Remove isolated nodes and their labels/strengths ---
        isolated_nodes = list(nx.isolates(G))
        for node in isolated_nodes:
            G.remove_node(node)
            node_labels.pop(node, None)
            node_strengths.pop(node, None)
        # -------------------------------------------------------

    # Compute levels and arrange nodes with more space
    levels = compute_levels(G)
    max_level = max(levels.values()) if levels else 0
    level_nodes = defaultdict(list)
    for node, lvl in levels.items():
        level_nodes[lvl].append(node)
    pos = {}
    horizontal_spacing = 10  # More horizontal space between nodes
    vertical_spacing = 3  # More vertical space between levels
    for lvl in range(max_level + 1):
        nodes = level_nodes[lvl]
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) * horizontal_spacing if n >= 1 else 0
            y = -lvl * vertical_spacing
            pos[node] = (x, y)

    # Increase figure size for large graphs
    plt.figure(figsize=(30, 10))

    # Node visual properties based on strength
    min_node_size = 200
    max_node_size = 2000
    min_alpha = 0.01
    max_alpha = 1.0
    strengths = [node_strengths[n] for n in G.nodes]
    node_sizes = [
        min_node_size + (max_node_size - min_node_size) * s
        for s in strengths
    ]
    node_alphas = [
        min_alpha + (max_alpha - min_alpha) * s
        for s in strengths
    ]

    # Draw nodes one by one to set individual alpha
    for node, size, alpha in zip(G.nodes, node_sizes, node_alphas):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color='lightblue',
            node_size=size,
            alpha=alpha
        )

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    # Draw edges with different styles, width and alpha
    min_width = 1
    max_width = 4
    min_edge_alpha = 0.01
    max_edge_alpha = 1.0
    for (u, v), style in edge_styles.items():
        weight = style['weight']
        width = min_width + (max_width - min_width) * weight
        alpha = min_edge_alpha + (max_edge_alpha - min_edge_alpha) * weight
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            edge_color=style['color'],
            style=style['style'],
            width=width,
            alpha=alpha,
            arrows=True,
            arrowstyle='-|>',
            min_source_margin=23,
            min_target_margin=23,
        )

    plt.axis('off')
    plt.tight_layout()
    plt.show()
    input("\nPress Enter to continue...")




class Argument:
    def __init__(self, attr_name, attr_value):
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.id = attr_name + str(attr_value)
        self.current_w = 1

    def get_id(self):
        return self.id

    def get_attr_name(self):
        return self.attr_name

    def get_attr_value(self):
        return self.attr_value

    def get_current_w(self):
        return self.current_w

    def set_current_w(self, new_w):
        self.current_w = new_w

    def __str__(self):
        return self.attr_name + "=" + str(self.attr_value)

    def __repr__(self):
        return self.attr_name + "=" + str(self.attr_value)


class Graph:
    def __init__(self):
        # dictionary of arguments -> weight of argument
        self.args = []
        # dictionary of (attacker, attacked) -> strength of attack
        self.attacks = dict()
        # initial weight of arguments
        self.initial_weight = 1
        # Shapley values
        self.shapley_values = None


    def get_initial_weight(self):
        return self.initial_weight

    def add_arg(self, name, value):
        arg = Argument(name, value)
        self.args.append(arg)


    def get_arg(self, id):
        for a in self.args:
            if a.get_id() == id:
                return a

    def set_contributions(self, contributions:dict):
        self.contributions = contributions

    def get_contribution(self, feature_name):
        return self.contributions[feature_name]

    def get_contributions(self):
        return self.contributions

    def get_args(self):
        return self.args

    def add_att(self, attacker, attacked, weight=None):
        if (attacker, attacked) not in self.attacks:
            self.attacks[(attacker, attacked)] = 1 if weight is None else weight
        else:
            if weight is None:
                self.attacks[(attacker, attacked)] += 1
            else:
                self.attacks[(attacker, attacked)] += weight

    def get_attacks(self):
        return self.attacks

    def get_attacked(self, attacked):
        attacks = self.attacks
        attackers = {}
        for att in attacks:
            if att[1] == attacked:
                attackers[att]=attacks[att]
        return attackers

    def set_attacks(self, attacks):
        self.attacks = attacks

    def get_attackers(self):
        args = self.get_args()
        attacks = self.get_attacks()
        arguments = []
        attackers = []

        for arg in args:
            temp = []
            for att in attacks.keys():
                if att[1] == arg:
                    temp.append(att[0])
            attackers.append(temp)
            arguments.append(arg)

        return arguments, attackers

    def update_arg_strength(self, arg, weight):
        arg.set_current_w(weight)

    def print_args(self, debug_path=None):
        for a in self.args:
            fprint((str(a) + " with weight: " + str(a.get_current_w())), debug_path)
        # print()

    def print_attacks(self, debug_path=None):
        for a in self.attacks.keys():
            fprint((str(a[0]) + " attacks ", str(a[1]) + " with strength " + str(self.attacks[a])), debug_path)


def get_attr_values(data, attr_name):
    return data[attr_name].unique()

def encode(df, dataset=None):
    if dataset == 'adult':
        cat_columns = ['age', 'workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country']
    if dataset == 'synthetic':
        cat_columns = ['age', 'workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                       'native-country']

    # Use pd.get_dummies() to create dummy variables
    dummies_df = pd.get_dummies(df, columns=cat_columns)
    return dummies_df

def create_arguments(data):
    data = data.iloc[:, :-1]

    for d in data:
        attr_values = get_attr_values(data, d)
        for v in attr_values:
            graph.add_arg(d, v)

def count_category_values(df, column_name):
    """
    Counts occurrences of each unique value in a categorical column of a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the categorical column.

    Returns:
        dict: A dictionary where keys are unique values from the column, and values are their counts.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    return df[column_name].value_counts().to_dict()

def add_attacks_v1(data):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]

    for i in range(len(neighbours)):
        n = neighbours.iloc[i, :]
        if n[-1:].values != individual[-1:].values:
            for d in data:
                n_data = n[:-1]
                ind_data = individual[:-1]
                value = n_data[d]
                attacker = graph.get_arg(d + str(value))
                attacked = graph.get_arg(d + str(ind_data[d]))
                if n_data[d] != ind_data[d]:
                    graph.add_att(attacker, attacked, 0.01)
                else:
                    graph.add_att(attacker, attacked, -0.01)


def count_value_not_occurred(df, column_name, value):
    """
    Counts the number of times a specific value does not appear in a column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    column_name (str): The name of the column to search.
    value: The value to count.

    Returns:
    int: The count of occurrences of the value in the column.
    """
    if column_name in df.columns:
        return (df[column_name] != value).sum()
    else:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")


def count_value_occurances(df, column_name, value, negative_label=">50K", use_labels = False):
    """
    Counts the number of times a specific value appears in a column of a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to search.
    column_name (str): The name of the column to search.
    value: The value to count.

    Returns:
    int: The count of occurrences of the value in the column.
    """
    if column_name in df.columns:
        if use_labels:
            return ((df[column_name] == value) & (df.iloc[:,-1]!=negative_label)).sum()
        else:
            return (df[column_name] == value).sum()
    else:
        raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")


def add_attacks2(data, k):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]

    for i in range(len(neighbours)):
        n = neighbours.iloc[i, :]
        for d in data:
            n_data = n[:-1]
            ind_data = individual[:-1]
            if n[-1:].values != individual[-1:].values: #P(f_n != f_ind | C_n != C_ind) > P(f_n == f_ind| C_n != C_ind)
                value = n_data[d]
                attacker = graph.get_arg(d + str(value))
                attacked = graph.get_arg(d + str(ind_data[d]))

                if n_data[d] != ind_data[d]:
                    denominator = count_value_not_occurred(data, d, ind_data[d])
                    # graph.add_att(attacker, attacked, ((0.2 *len(neighbours))/denominator))
                    graph.add_att(attacker, attacked, 1/denominator)
                else:
                    denominator = count_value_occurances(data, d, ind_data[d])
                    # graph.add_att(attacker, attacked, (-(0.2 *len(neighbours)) / denominator))
                    graph.add_att(attacker, attacked, -1/denominator)
            # else:
            #     if n_data[d] == ind_data[d]:
            #         graph.add_att(attacker, attacked, -0.01)

def add_attacks(data):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]

    for i in range(len(neighbours)):
        n = neighbours.iloc[i, :]
        for d in data:
            n_data = n[:-1]
            ind_data = individual[:-1]
            value = n_data[d]
            attacker = graph.get_arg(d + str(value))
            attacked = graph.get_arg(d + str(ind_data[d]))

            if n_data[d] != ind_data[d]: # P(C_n != C_ind | f_n != f_ind) > P(C_n == C_ind | f_n != f_ind)
                if n[-1:].values != individual[-1:].values:
                    graph.add_att(attacker, attacked, 0.01)
                else:
                    graph.add_att(attacker, attacked, -0.01)
            # else:
            #     if n[-1:].values == individual[-1:].values:
            #         graph.add_att(attacker, attacked, -0.01)


    # for i in range(len(neighbours)):
    #     n = neighbours.iloc[i, :]
    #     for d in data:
    #         n_data = n[:-1]
    #         ind_data = individual[:-1]
    #         value = n_data[d]
    #         attacker = graph.get_arg(d + str(value))
    #         attacked = graph.get_arg(d + str(ind_data[d]))
    #         attack_list = graph.get_attacked(attacked)
    #         if n[-1:].values != individual[-1:].values:
    #             if n_data[d] == ind_data[d]: # P(f_n != f_ind | C_n != C_ind) > P(f_n == f_ind| C_n != C_ind)
    #                 # for attack in attack_list:
    #                     # graph.add_att(attacker, attack[0], 0.01)
    #                     # graph.add_att(attack[0], attacker, 0.01)
    #                 graph.add_att(attacker, attacked, -0.01)
    #             # commented out since it has already been computed in the previous if statement
    #             # else:
    #             #     graph.add_att(attacker, attacked, 0.01)



def add_attacks_global(data):
    for d in data:
        att_values = get_attr_values(data, d)
        att_counts = count_category_values(data, d)
        for v1 in att_values:
            for v2 in att_values:
                if v1 != v2:
                    attacker = graph.get_arg(d + str(v1))
                    attacked = graph.get_arg(d + str(v2))
                    count_diff = att_counts[v1] - att_counts[v2]
                    attack_weight = 0.01 * count_diff
                    graph.add_att(attacker, attacked, attack_weight)




    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]

    for i in range(len(neighbours)):
        n = neighbours.iloc[i, :]
        for d in data:
            n_data = n[:-1]
            ind_data = individual[:-1]
            value = n_data[d]
            attacker = graph.get_arg(d + str(value))
            attacked = graph.get_arg(d + str(ind_data[d]))

            if n[-1:].values != individual[-1:].values: #P(f_n != f_ind | C_n != C_ind) > P(f_n == f_ind| C_n != C_ind)
                if n_data[d] != ind_data[d]:
                    graph.add_att(attacker, attacked, 0.01)
                else:
                    graph.add_att(attacker, attacked, -0.01)





def add_attacks_oana(data):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]

    for i in range(len(neighbours)):
        n = neighbours.iloc[i, :]
        if n[-1:].values != individual[-1:].values:
            for d in data:
                n_data = n[:-1]
                ind_data = individual[:-1]
                if n_data[d] != ind_data[d]:
                    for column in n_data.index:
                        value = n_data[column]
                        attacker = graph.get_arg(column + str(value))
                        attacked = graph.get_arg(d + str(ind_data[d]))
                        graph.add_att(attacker, attacked)


def attack_strengths(k):
    attacks = graph.get_attacks()
    norm_attacks = dict()
    for a in attacks.keys():
        norm_attacks[a] = attacks.get(a) / k
    graph.set_attacks(norm_attacks)


def incoming_weight(attacker_weight: float, attack_strength: float) -> float:
    """
    Calculates the incoming weight of a single attacker to an argument
    @param attacker_weight: weight of a single incoming attacker (in interval [0,1])
    @param attack_strength: strength of the attack (in interval [0,1])
    @return: the weight of the attacker multiplied by the strength of the attack (in interval [0,1])
    """
    return attacker_weight * attack_strength


def aggregate(arg: object) -> float:
    """
    Calculates the aggregation of the incoming weights of attackers to an argument
    @param arg: the argument to calculate the aggregation of
    @return: the sum of all incoming weights to the argument multiplied by their respective attack strengths
    """
    arguments, attackers = graph.get_attackers()
    attack_strengths = graph.get_attacks()
    total = 0
    arg_index = arguments.index(arg)
    for a in attackers[arg_index]:
        total = total - incoming_weight(a.get_current_w(), attack_strengths.get((a, arg)))
    return total


def influence(arg: object) -> float:
    """
    Calculates the influence on an argument at a point in time, using the CAT semantics
    Weighted h-categorizer semantics (Hbs), Amgoud et al. 2022
    @param arg: the argument to update the weight of
    @return: the change in weight of the argument at a point in time
    """
    w = graph.get_initial_weight()
    update = w / (1 + (-aggregate(arg) ) )
    return update


def weight_diffs(new_weights, current_weights):
    epsilon = 0.01
    for i in range(len(new_weights)):
        if abs(new_weights[i] - current_weights[i]) > epsilon:
            return True
    return False


def calculate_final_weights():
    args = graph.get_args()
    current_weights = []
    for arg in args:
        arg.set_current_w(graph.get_initial_weight())
        current_weights.append(arg.get_current_w())

    not_converged = True
    # while the difference in changes is greater than epsilon (have not reached convergence threshold)
    while not_converged:
        diff_changes = []
        new_weights = []
        # for each argument, calculate the new weight and store
        for arg in args:
            new_weight = influence(arg)
            new_weights.append(new_weight)
        for i in range(len(new_weights)):
            graph.update_arg_strength(args[i], new_weights[i])

        # if any difference in weight changes is greater than epsilon, all arguments not converged
        not_converged = weight_diffs(new_weights, current_weights)

        # set current weights to new weights
        current_weights = new_weights


def display(instance, debug_path=None):
    fprint(str(instance.values), debug_path)
    graph.print_args(debug_path)
    fprint('', debug_path)
    graph.print_attacks(debug_path)
    fprint('-'*40, debug_path)
    # graph.print_args()
    # print()

def get_final_weights():
    final_weights = {}
    args = graph.get_args()
    for a in args:
        # add argument and its weight to dictionary
        final_weights[str(a)] = a.get_current_w()
    return final_weights


def get_weakest_args(final_weights, initial_weight = 1):
    # weakest_args = []
    weakest_args = []
    # find the weakest arguments
    max_weight = max(final_weights.values())
    if max_weight != initial_weight:
        for arg in final_weights.keys():
            if 'EBA' in arg:
                if final_weights[arg] > initial_weight:
                # if final_weights[arg] == min_weight:
                    # weakest_args.append(arg)
                    weakest_args.append(arg)
        return weakest_args
    else:
        return ["consistent"]

def get_weakest_args_super(final_weights, initial_weight = 1, epsilon = 0.01):
    # weakest_args = []
    weakest_args = []
    # find the weakest arguments
    max_weight = max(final_weights.values())
    if max_weight != initial_weight:
        for arg in final_weights.keys():
            if 'EBA' in arg and 'total' in arg:
                if final_weights[arg] - initial_weight > epsilon:
                # if final_weights[arg] == min_weight:
                    # weakest_args.append(arg)
                    weakest_args.append(arg)
        return weakest_args
    else:
        return ["consistent"]


def get_weakest_args_oana(final_weights):
    weakest_args = []
    # find the weakest arguments
    min_weight = min(final_weights.values())
    if min_weight != 1:
        for arg in final_weights.keys():
            if final_weights[arg] == min_weight:
                weakest_args.append(arg)
        return weakest_args
    else:
        return ["consistent"]


def visualize_argumentation_graph():
    """Visualizes the argumentation graph using NetworkX."""

    G = nx.DiGraph()
    for arg in graph.get_args():
      G.add_node(arg.get_id(), initial_weight=graph.get_initial_weight(), final_weight=arg.get_current_w())

    for (attacker, attacked), weight in graph.get_attacks().items():
      G.add_edge(attacker.get_id(), attacked.get_id(), weight=weight)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray', arrows=True)

    # Add initial and final weights to node labels
    node_labels = {}
    for node in G.nodes():
        node_labels[node] = f"{node}\nInitial: {G.nodes[node]['initial_weight']:.2f}\nFinal: {G.nodes[node]['final_weight']:.2f}"
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Add attack weights to edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Argumentation Graph")
    plt.show()




def construct_graph(inds, method, debug_path=None, mode=1, k=5):
    # inds = pd.read_csv(filename)


    # initialise graph
    global graph
    graph = Graph()

    # create arguments using Definition 2
    create_arguments(inds)

    # add attacks using Definition 4
    if method == methods.Our:
        if mode == 1:
            add_attacks(inds)
        elif mode == 2:
            add_attacks2(inds, k)
    else:
        add_attacks_oana(inds)

    # set attack strengths using Definition 4
    attack_strengths(len(inds) - 1)

    # calculate the final weights using Hbs semantics defined in Equation 1 and 2
    calculate_final_weights()

    # display the graph details (optional for testing)
    display(inds.iloc[0], debug_path)

    # get final weights of arguments to 2dp
    final_weights = get_final_weights()

    # get the weakest arguments of the graph
    if method == methods.Our:
        weakest_args = get_weakest_args(final_weights)
    elif method == methods.Oana:
        weakest_args = get_weakest_args_oana(final_weights)

    # visualize the argumentation framework with nodes as arguments and attacks as edges (option for testing)
    # visualize_argumentation_graph()

    return final_weights, weakest_args


def create_BAG_file(data, filename, neighborhood_properties=None, protected_features=None):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    data = data.iloc[:, :-1]
    # initial_score_individuals = 0.5
    # initial_score_neighbours = 1
    # with open(filename, 'w') as f:
    #     # Adding Arguments
    #     for d in data:
    #         attr_values = get_attr_values(data, d)
    #         individual_attr_value = individual[d]
    #         for v in attr_values:
    #             if individual_attr_value == v:
    #                 f.write(f"arg({d}:{v}, {initial_score_individuals})\n")
    #             else:
    #                 f.write(f"arg({d}:{v}, {initial_score_neighbours})\n")
    #
    #     # Adding attacks and supports
    #     for i in range(len(neighbours)):
    #         n = neighbours.iloc[i, :]
    #         for d in data:
    #             n_data = n[:-1]
    #             ind_data = individual[:-1]
    #             if n[-1:].values != individual[
    #                                 -1:].values:  # P(f_n != f_ind | C_n != C_ind) > P(f_n == f_ind| C_n != C_ind)
    #                 value = n_data[d]
    #
    #                 if n_data[d] != ind_data[d]:
    #                     denominator = count_value_not_occurred(data, d, ind_data[d])
    #                     f.write(f"att({d}:{value},{d}:{ind_data[d]}, {1 / denominator})\n")
    #                 else:
    #                     denominator = count_value_occurances(data, d, value)
    #                     f.write(f"sup({d}:{value},{d}:{ind_data[d]}, {1 / denominator})\n")

    significance_threshold = 30
    diversity_threshold = 0.1
    if neighborhood_properties is not None:
        diversity_attack_weights = {}
        for property, value in neighborhood_properties.items():
            if 'diversity' in property:
                    diversity_attack_weights[property] = (diversity_threshold - value)/diversity_threshold if value < diversity_threshold else 0
        significance_attack_weight = (significance_threshold - neighborhood_properties['N-significance'])/significance_threshold if neighborhood_properties['N-significance'] < significance_threshold else 0
        s_objective_attack_weight = 0 if neighborhood_properties['S-Objective'] else 1



    initial_score_feature = 0.5
    initial_score_values = 1
    with open(filename, 'w') as f:
        # Adding Arguments
        for d in data:
            if protected_features is not None:
                if d not in protected_features:
                    continue
            individual_attr_value = individual[d]
            f.write(f"arg({d}:EBA-{individual_attr_value}, {initial_score_feature})\n")
            attr_values = get_attr_values(data, d)
            for v in attr_values:
                f.write(f"arg({d}:{v}, {initial_score_values})\n")
            if neighborhood_properties is not None:
                f.write(f"arg(NotDiverse:{d}, {initial_score_values})\n")
        if neighborhood_properties is not None:
            f.write(f"arg(NotSignificant, {initial_score_values})\n")
            f.write(f"arg(NotObjective, {initial_score_values})\n")


        # Adding attacks and supports
        attacks = {}
        supports = {}
        for i in range(len(neighbours)):
            n = neighbours.iloc[i, :]
            for d in data:
                if protected_features is not None:
                    if d not in protected_features:
                        continue
                n_data = n[:-1]
                ind_data = individual[:-1]
                if n[-1:].values != individual[
                                    -1:].values:  # P( C_n != C_ind | f_n != f_ind ) > P( C_n != C_ind | f_n == f_ind)
                    value = n_data[d]

                    if n_data[d] != ind_data[d]:
                        denominator = count_value_not_occurred(data, d, ind_data[d])
                        key = f'sup({d}:{value},{d}:EBA-{ind_data[d]}'
                        value = (1 / denominator) + (supports[key] if key in supports.keys() else 0)
                        supports[key] = value
                        # f.write(f"att({d}:{value},{d}:EBA-{ind_data[d]}, {1/denominator})\n")
                    else:
                        denominator = count_value_occurances(data, d, value)
                        key = f'att({d}:{value},{d}:EBA-{ind_data[d]}'
                        value = (1 / denominator) + (attacks[key] if key in attacks.keys() else 0)
                        attacks[key] = value
                        # f.write(f"sup({d}:{value},{d}:EBA-{ind_data[d]}, {1/denominator})\n")


        if neighborhood_properties is not None:
            for d in data:
                if protected_features is not None:
                    if d not in protected_features:
                        continue
                    ind_data = individual[:-1]
                    key = f'att(NotDiverse:{d},{d}:EBA-{ind_data[d]}'
                    value = diversity_attack_weights[f'diversity_{d}']
                    attacks[key] = value
                    key = f'att(NotSignificant,{d}:EBA-{ind_data[d]}'
                    value = significance_attack_weight
                    attacks[key] = value
                    key = f'att(NotObjective,{d}:EBA-{ind_data[d]}'
                    value = s_objective_attack_weight
                    attacks[key] = value


        for key,value in attacks.items():
            f.write(f"{key}, {value})\n")
        for key,value in supports.items():
            f.write(f"{key}, {value})\n")


def compute_initial_weight(data, feature_name, feature_value, individual_value, nagative_label='<=50K'):
    feature_value_occurances = count_value_occurances(data, feature_name, feature_value, nagative_label, use_labels=True)
    if feature_value == individual_value:
        individual_value_occurances = count_value_occurances(data, feature_name, feature_value)
    else:
        individual_value_occurances = count_value_not_occurred(data, feature_name, individual_value)
    return feature_value_occurances/individual_value_occurances
def create_BAG_file_no_relation_weight(data, filename, neighborhood_properties=None, protected_features=None):
    individual = data.iloc[0, :]
    neighbours = data.iloc[1:, :]
    labeled_data = copy.deepcopy(data)
    data = data.iloc[:, :-1]

    significance_threshold = 30
    diversity_threshold = 0.2
    if neighborhood_properties is not None:
        diversity_weights = {}
        for property, value in neighborhood_properties.items():
            if 'diversity' in property:
                    diversity_weights[property] = (diversity_threshold - value)/diversity_threshold if value < diversity_threshold else 0
        significance_weight = (significance_threshold - neighborhood_properties['N-significance'])/significance_threshold if neighborhood_properties['N-significance'] < significance_threshold else 0
        s_objective_weight = 0 if neighborhood_properties['S-Objective'] else 1



    initial_score_feature = 0
    with open(filename, 'w') as f:
        # Adding Arguments and Attack and Support Relations
        for d in data:
            if protected_features is not None:
                if d not in protected_features:
                    continue
            individual_feature_value = individual[d]
            f.write(f"arg({d}:EBA-{individual_feature_value}-total, {initial_score_feature})\n")
            feature_values = get_attr_values(data, d)
            for feature_value in feature_values:
                initial_score_value = compute_initial_weight(labeled_data, d, feature_value, individual_feature_value)
                f.write(f"arg({d}:{feature_value}, {initial_score_value})\n")
                if feature_value == individual_feature_value:
                    f.write(f'att({d}:{feature_value},{d}:EBA-{individual_feature_value}-total)\n')
                else:
                    f.write(f'sup({d}:{feature_value},{d}:EBA-{individual_feature_value}-total)\n')

            if neighborhood_properties is not None:
                f.write(f"arg(NotDiverse:{d}, {diversity_weights[f'diversity_{d}']})\n")
        if neighborhood_properties is not None:
            f.write(f"arg(NotSignificant, {significance_weight})\n")
            f.write(f"arg(NotObjective, {s_objective_weight})\n")


        # # Adding attacks and supports
        # for i in range(len(neighbours)):
        #     n = neighbours.iloc[i, :]
        #     for d in data:
        #         if protected_features is not None:
        #             if d not in protected_features:
        #                 continue
        #         n_data = n[:-1]
        #         ind_data = individual[:-1]
        #         if n[-1:].values != individual[
        #                             -1:].values:  # P( C_n != C_ind | f_n != f_ind ) > P( C_n != C_ind | f_n == f_ind)
        #             value = n_data[d]
        #
        #             if n_data[d] != ind_data[d]:
        #                 f.write(f'sup({d}:{value},{d}:EBA-{ind_data[d]})')
        #             else:
        #                 f.write(f'att({d}:{value},{d}:EBA-{ind_data[d]})')


        if neighborhood_properties is not None:
            for d in data:
                if protected_features is not None:
                    if d not in protected_features:
                        continue
                    ind_data = individual[:-1]
                    f.write(f'att(NotDiverse:{d},{d}:EBA-{ind_data[d]}-total)')
                    f.write(f'att(NotSignificant,{d}:EBA-{ind_data[d]}-total)')
                    f.write(f'att(NotObjective,{d}:EBA-{ind_data[d]}-total)')

def create_BAG_file_no_relation_weight_multi(neighborhood_dict, filename, protected_features, negative_reasoning_path=True, negative_label='<=50K', diversity_threshold=0.2, significance_threshold=30):
    # Create top level argument for each feature aggregating all the supports
    # from the neighbourhood-level argumentation frameworks
    initial_score_feature = 0
    with open(filename, 'w') as f:
        for d in protected_features:
            key = list(neighborhood_dict.keys())[0]
            individual = neighborhood_dict[key]['neighborhoods'].iloc[0, :]
            individual_feature_value = individual[d]
            f.write(f"arg({d}:EBA-{individual_feature_value}-total, {initial_score_feature})\n") # Top-level argument of Epsilon Biased Against (EBA)

    for k, dict in neighborhood_dict.items():
        data = dict["neighborhoods"]
        neighborhood_properties = dict["neighborhood_properties"]
        individual = data.iloc[0, :]
        neighbours = data.iloc[1:, :]
        labeled_data = copy.deepcopy(data)
        data = data.iloc[:, :-1]

        if neighborhood_properties is not None:
            diversity_weights = {}
            for property, value in neighborhood_properties.items():
                if 'diversity' in property:
                        diversity_weights[property] = (diversity_threshold - value)/diversity_threshold if value < diversity_threshold else 0
            significance_weight = (significance_threshold - neighborhood_properties['N-significance'])/significance_threshold if neighborhood_properties['N-significance'] < significance_threshold else 0
            s_objective_weight = 0 if neighborhood_properties['S-Objective'] else 1



        initial_score_feature = 0
        with open(filename, 'a') as f:
            # Adding Arguments and Attack and Support Relations
            for d in protected_features:
                individual_feature_value = individual[d]
                f.write(f"arg({d}:EBA-{individual_feature_value}-{k}, {initial_score_feature})\n")
                if negative_reasoning_path:
                    f.write(f"arg({d}:NEBA-{individual_feature_value}-{k}, {initial_score_feature})\n") # Not Epsilon Biased Against (NEBA) Argument
                feature_values = get_attr_values(data, d)
                for feature_value in feature_values:
                    initial_score_value = compute_initial_weight(labeled_data, d, feature_value, individual_feature_value, negative_label)
                    f.write(f"arg({d}:{feature_value}-{k}, {initial_score_value})\n")
                    if feature_value == individual_feature_value:
                        f.write(f'att({d}:{feature_value}-{k},{d}:EBA-{individual_feature_value}-{k})\n')
                        if negative_reasoning_path:
                            f.write(f'sup({d}:{feature_value}-{k},{d}:NEBA-{individual_feature_value}-{k})\n')
                    else:
                        f.write(f'sup({d}:{feature_value}-{k},{d}:EBA-{individual_feature_value}-{k})\n')
                        if negative_reasoning_path:
                            f.write(f'att({d}:{feature_value}-{k},{d}:NEBA-{individual_feature_value}-{k})\n')

                if neighborhood_properties is not None:
                    f.write(f"arg(NotDiverse-{k}:{d}, {diversity_weights[f'diversity_{d}']})\n")
            if neighborhood_properties is not None:
                f.write(f"arg(NotSignificant-{k}, {significance_weight})\n")
                f.write(f"arg(NotObjective-{k}, {s_objective_weight})\n")


            if neighborhood_properties is not None:
                for d in protected_features:
                    ind_data = individual[:-1]
                    f.write(f'att(NotDiverse-{k}:{d},{d}:EBA-{ind_data[d]}-{k})\n')
                    f.write(f'att(NotSignificant-{k},{d}:EBA-{ind_data[d]}-{k})\n')
                    f.write(f'att(NotObjective-{k},{d}:EBA-{ind_data[d]}-{k})\n')
                    if negative_reasoning_path:
                        f.write(f'att(NotDiverse-{k}:{d},{d}:NEBA-{ind_data[d]}-{k})\n')
                        f.write(f'att(NotSignificant-{k},{d}:NEBA-{ind_data[d]}-{k})\n')
                        f.write(f'att(NotObjective-{k},{d}:NEBA-{ind_data[d]}-{k})\n')
            for d in protected_features:
                individual_feature_value = individual[d]
                # Final support relations from the neighborhood-level argumentation frameworks to the top level argument
                f.write(f'sup({d}:EBA-{individual_feature_value}-{k}, {d}:EBA-{individual_feature_value}-total)\n')
                if negative_reasoning_path:
                    f.write(f'att({d}:NEBA-{individual_feature_value}-{k}, {d}:EBA-{individual_feature_value}-total)\n')





def extract_strength_of_individual(model, individual_initial_strength, protected_attrs=None):
    all_strengths = model.argument_strength
    indv_strengths = {}
    for arg, strength in all_strengths.items():
        # if individual_initial_strength == arg.initial_weight:
            if protected_attrs is None:
                indv_strengths[arg.name] = strength
            elif arg.name.split(':')[0] in protected_attrs:
                indv_strengths[arg.name] = strength

    return indv_strengths


def convert_colon_to_eq(items):
    newItems = []
    for item in items:
        newItems.append(item.replace(':', '=').strip())
    return newItems


def construct_graph_QE(inds, method,  debug_path=None, neighborhood_properties=None, mode=1, k=5, protected_attrs = ['race', 'sex']):
    import sys
    sys.path.append("Uncertainpy/src")
    individual_initial_strength = 0.0
    import uncertainpy.gradual as grad
    bag_filename = "BAG.bag"
    # Define your model
    model = grad.semantics.ContinuousModularModel(grad.semantics.modular.SumAggregation(),
                                                  grad.semantics.modular.QuadraticMaximumInfluence(conservativeness=1))
    # Set an approximator
    model.approximator = grad.algorithms.RK4(model)
    create_BAG_file_no_relation_weight(inds, bag_filename, neighborhood_properties, protected_attrs)
    # Set the BAG
    model.BAG = grad.BAG(bag_filename)

    model.solve(delta=10e-2, epsilon=10e-4, verbose=False, generate_plot=False)

    individual_final_strengths = extract_strength_of_individual(model, individual_initial_strength, protected_attrs)

    # get the weakest arguments of the graph
    weakest_args = get_weakest_args(individual_final_strengths, initial_weight=individual_initial_strength)
    weakest_args = convert_colon_to_eq(weakest_args)

    return individual_final_strengths, weakest_args


def construct_graph_QE_multi(neighborhoods_dict, method,  debug_path=None, mode=1, k=5,
                             protected_attrs = ['race', 'sex'], plot_argumentation_framework = False,
                             negative_label='<=50K', epsilon=0.01, diversity_threshold=0.2, significance_threshold=30,
                             negative_reasoning_path = False):
    import sys
    sys.path.append("Uncertainpy/src")
    individual_initial_strength = 0.0
    import uncertainpy.gradual as grad
    bag_filename = "BAG.bag"
    # Define your model
    model = grad.semantics.ContinuousModularModel(grad.semantics.modular.SumAggregation(),
                                                  grad.semantics.modular.QuadraticMaximumInfluence(conservativeness=1))
    # Set an approximator
    model.approximator = grad.algorithms.RK4(model)
    create_BAG_file_no_relation_weight_multi(neighborhoods_dict, bag_filename, protected_attrs,
                                             negative_label = negative_label,
                                             diversity_threshold = diversity_threshold,
                                             significance_threshold = significance_threshold,
                                             negative_reasoning_path = negative_reasoning_path)
    # Set the BAG
    model.BAG = grad.BAG(bag_filename)

    model.solve(delta=10e-2, epsilon=10e-4, verbose=False, generate_plot=False)

    individual_final_strengths = extract_strength_of_individual(model, individual_initial_strength, protected_attrs)

    # get the weakest arguments of the graph
    weakest_args = get_weakest_args_super(individual_final_strengths, initial_weight=individual_initial_strength, epsilon = epsilon)
    weakest_args = convert_colon_to_eq(weakest_args)

    if plot_argumentation_framework and len(weakest_args)!=0 and weakest_args != ['consistent']:
        plot_hierarchical_arguments(model)

    return individual_final_strengths, weakest_args

def construct_global_graph(data):
    protected_features = ['sex', 'race', 'class']
    data = data[protected_features]
    global graph
    graph = Graph()

    # create arguments using Definition 2
    create_arguments(data)


    add_attacks_global(data.iloc[:, :-1])

    # set attack strengths using Definition 4
    attack_strengths(len(data) - 1)

    # calculate the final weights using Hbs semantics defined in Equation 1 and 2
    calculate_final_weights()

    # display the graph details (optional for testing)
    display(data[0])

    # get final weights of arguments to 2dp
    final_weights = get_final_weights()

    # get the weakest arguments of the graph
    weakest_args = get_weakest_args(final_weights)

    # visualize_argumentation_graph()

    return graph, weakest_args
