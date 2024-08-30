import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# (Create Basic GraphSetting) #####################################################
class GenerateNodeMap():
    def __init__(self, n_nodes, distance_scale=1, random_state=None):
        self.n_nodes = n_nodes
        self.centers = np.zeros((n_nodes,2))
        self.covs = np.ones(n_nodes)
        self.adj_base_matrix = np.zeros((n_nodes, n_nodes))
        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        self.adj_dist_mat = np.zeros((n_nodes, n_nodes))

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.distance_scale = distance_scale
    
    # (Version 3.3)
    def generate_nodes(self, n_nodes):
        centers_x = self.rng.uniform(0,1, size=n_nodes) 
        centers_y = self.rng.uniform(0,1, size=n_nodes) 
        centers = np.stack([centers_x, centers_y]).T
        return centers

    # (Version 3.3)
    def make_adjacent_matrix(self, centers, scale=50):
        adj_matrix = np.zeros((len(centers), len(centers)))
        adj_dist_mat = np.zeros((len(centers), len(centers)))

        # node기반 adjacent matrix 구성
        # scale = 50
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                distance = np.sqrt((centers[i][0] - centers[j][0])**2 +
                                (centers[i][1] - centers[j][1])**2)
                # adj_matrix[i][j] = distance * self.rng.uniform(1, 3) * scale
                # adj_matrix[j][i] = distance * self.rng.uniform(1, 3) * scale
                adj_matrix[i][j] = distance * self.rng.normal(scale, 10/10)
                adj_matrix[j][i] = distance * self.rng.normal(scale, 10/10)
                
                adj_dist_mat[i][j] = distance
                adj_dist_mat[j][i] = distance
        return (adj_matrix, adj_dist_mat)

    # (Version 3.3)
    def verify_closeness(self, adj_dist_mat, n_nodes, criteria_scale=0.25):
        adj_dist_mat_copy = adj_dist_mat.copy()
        np.fill_diagonal(adj_dist_mat_copy, np.inf)
        d = np.sqrt(2) / np.sqrt(n_nodes)

        near_points = (adj_dist_mat_copy < d*criteria_scale).sum(1).astype(bool)
        # print(near_points.sum())
        return near_points

    # (Version 3.3)
    def create_node(self, node_scale=50, cov_knn=3):
        # (Version 3.3) -------------------------------------------------------------
        n_nodes = self.n_nodes
        centers = self.generate_nodes(n_nodes)
        adj_base_matrix, adj_dist_mat = self.make_adjacent_matrix(centers, scale=node_scale)
        v_closeness = self.verify_closeness(adj_dist_mat, n_nodes)

        centers = centers[~v_closeness,:]

        it = 0
        while(len(centers) < n_nodes):
            # print(it, end=" ")
            new_centers = self.generate_nodes(n_nodes - len(centers))
            centers = np.append(centers, new_centers, axis=0)
            adj_base_matrix, adj_dist_mat = self.make_adjacent_matrix(centers, scale=node_scale)
            v_closeness = self.verify_closeness(adj_dist_mat, n_nodes)
            centers = centers[~v_closeness,:]
            it +=1
            if it >= 100:
                raise Exception("need to be lower 'criteria_scale' in 'verify_closeness' function.")
        # print(it)

        self.centers = centers[np.argsort((centers**2).sum(1))]        # sorted
        self.adj_base_matrix, self.adj_dist_mat = self.make_adjacent_matrix(self.centers, scale=node_scale)

        # Covariance of Nodes
        adj_dist_mat_copy = self.adj_dist_mat.copy()
        np.fill_diagonal(adj_dist_mat_copy, np.inf)
        adj_dist_mat_rank = matrix_rank(adj_dist_mat_copy, axis=1)

        adj_nearnode = (adj_dist_mat_copy * (adj_dist_mat_rank <= cov_knn))
        np.fill_diagonal(adj_nearnode, 0)

        self.covs = adj_nearnode.mean(1) * (n_nodes/10)*2 * (3/cov_knn)**(1.3)
        # -----------------------------------------------------------------------------

    # (Version 3.3)
    def create_connect(self, connect_scale=0.13):
        if self.centers.sum() == 0:
            print("nodes are not created.")
        else:
            # (Version 3.3) -------------------------------------------------------------
            # assign probability for connection
            def connect_function(x, connect_scale=0):
                if x < 5:
                    x_con = (1/(x - connect_scale))
                elif x < 10:
                    x_con = (1/x)**(1.5 - connect_scale) 
                elif x < 20:
                    x_con = (1/x)**(2 - connect_scale) 
                else:
                    x_con = (1/x)**(5 - connect_scale) 
                return x_con
            
            n_nodes = self.n_nodes
            connect_scale = connect_scale

            # 모든 노드가 연결될때까지 재구성
            while(True):
                shape_mat = self.adj_base_matrix.shape
                adj_sym_mat = (self.adj_base_matrix + self.adj_base_matrix.T)/2
                np.fill_diagonal(adj_sym_mat, np.inf)
                adj_sym_rank_mat = (np.argsort(np.argsort(adj_sym_mat, axis=1), axis=1) + 1)

                adj_sym_rank_rev_mat = apply_function_to_matrix(adj_sym_rank_mat, connect_function, connect_scale=connect_scale)

                adj_sym_noise_mat = adj_sym_rank_rev_mat + self.rng.normal(0,0.01, size=shape_mat)
                np.fill_diagonal(adj_sym_noise_mat,0)

                adj_conn_prob_mat = adj_sym_noise_mat.copy()
                adj_sym_conn_prob_mat = (adj_conn_prob_mat.T + adj_conn_prob_mat)/2
                adj_sym_conn_prob_mat[adj_sym_conn_prob_mat<0] = 0
                adj_sym_conn_prob_mat[np.arange(shape_mat[0]),np.argmax(adj_sym_conn_prob_mat,axis=1)] = 1      # 최소 1개의 node끼리는 connect
                adj_sym_conn_prob_mat[adj_sym_conn_prob_mat>1] = 1

                connect_filter = (self.rng.binomial(1, adj_sym_conn_prob_mat)).astype(bool)
                connect_filter_sym = connect_filter.T + connect_filter

                if self.is_connected(connect_filter_sym):   # 모든 노드가 연결될 경우
                    print("connect ratio : ", connect_filter_sym.sum() / len(connect_filter_sym.ravel()) )
                    adj_matrix_con = self.adj_base_matrix * connect_filter_sym    # connect filtering
                    self.adj_matrix = adj_matrix_con
                    break
                else:
                    if connect_scale < 0.99:
                        connect_scale += 0.01
            # -----------------------------------------------------------------------------

    def is_connected(self, matrix, start=0):
        n = len(matrix)
        visited = np.zeros(n, dtype=bool)

        def dfs(node):
            stack = [node]
            while stack:
                current = stack.pop()
                if not visited[current]:
                    visited[current] = True
                    neighbors = np.where(matrix[current] == 1)[0]
                    for neighbor in neighbors:
                        if not visited[neighbor]:
                            stack.append(neighbor)
        dfs(start)

        # 모든 노드가 방문되었는지 확인
        return np.all(visited)

    def predict(self, x):
        # # multivariate guassian base
        # class_prob = np.array([gaussian_instance.pdf(point) for gaussian_instance in self.gaussian_objects])
        # maxprob_class = np.argmax(class_prob)       # class

        # distance base
        class_prob = 1/ (np.sqrt((self.centers - x)**2).sum(1) / self.covs)
        class_argmax = np.argmax(class_prob)       # class
        return class_prob, class_argmax
    
    def nearest_distance(self, x, distance_scale=None):
        class_prob, class_argmax = self.predict(x)
        distance_scale = self.distance_scale if distance_scale is None else distance_scale
        return np.sqrt(((self.centers[class_argmax] - np.array(x))**2).sum()) * distance_scale

    def __call__(self, x):
        return self.nearest_distance(x)




# visualize_graph -------------------------------------------------------------------------
def visualize_graph(centers, adjacent_matrix=None, path=[], path_distance=[], distance=None,
            covs=None,  point=None, class_prob=None, class_argmax=None, point_from=None, weight_direction='both', weight_vis_base_mat=None,
            title=None, vmax=3, return_plot=False):
    """ 
        nodes : nodes coordinate
        weight_direction : 'both', 'forward', 'backward'
    """
    # plot
    fig, ax = plt.subplots(figsize=(20,15))

    if title is not None:
        ax.set_title(title, fontsize=25)
    else:
        if len(path) > 0:
            if distance is None:
                ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n {path}", fontsize=25)
            else:
                if len(path_distance)>0:
                    sum_path_distance = np.sum(path_distance[:len(path)])
                    ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n (Dist: {sum_path_distance:.1f}) {path}", fontsize=25)
                else:
                    ax.set_title(f"The Shortest-Path from {path[0]} to {path[-1]} \n (Dist: {distance:.1f}) {path}", fontsize=25)
        else:
            ax.set_title("Graph Visualization from Adjacency Matrix", fontsize=25)


    if weight_vis_base_mat is not None:
        line_cmap = plt.get_cmap("coolwarm")
        # line_norm = plt.Normalize(1, 3)
        line_norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=line_cmap, norm=line_norm)
        plt.colorbar(sm, ax=ax)

    # 엣지 그리기
    for i in range(len(centers)):
        # draw edge ---------------------------------------------------------------------------
        if adjacent_matrix is not None:
            for j in range(i+1, len(adjacent_matrix)):
                if adjacent_matrix[i][j] > 0:
                    u = [centers[i][0], centers[j][0]]
                    v = [centers[i][1], centers[j][1]]

                    if weight_vis_base_mat is None:
                        ax.plot(u, v, color='gray', alpha=0.2)
                    else:
                        if weight_direction == 'both':
                            line_weight = (adjacent_matrix[i][j] + adjacent_matrix[j][i])/(weight_vis_base_mat[i][j] + weight_vis_base_mat[j][i])
                        elif weight_direction == 'forward':
                            line_weight = adjacent_matrix[i][j] / weight_vis_base_mat[i][j]
                        elif weight_direction == 'backward':
                            line_weight = adjacent_matrix[j][i] / weight_vis_base_mat[j][i]
                        
                        ax.plot(np.array(u)-0.005, np.array(v)-0.005, 
                            color=line_cmap(line_norm( line_weight )), alpha=0.5)
                        # ax.plot(np.array(u)-0.005, np.array(v)-0.005, 
                        #         color=line_cmap(line_norm(adjacent_matrix[i][j])), alpha=0.5)
                        # ax.plot(np.array(u)+0.005, np.array(v)+0.005,
                        #         color=line_cmap(line_norm(adjacent_matrix[j][i])), alpha=0.5)

        # draw node ---------------------------------------------------------------------------
        node = i

        if (point is not None) and (class_prob is not None) and (class_argmax is not None):
            # node_color = 'blue' if i == class_argmax else 'steelblue'
            node_linewidth = 1 if i == class_argmax else 0.3
            
            ax.scatter(centers[i][0], centers[i][1], label=f'Node {node}', color='skyblue', s=500, edgecolor='blue',
                        alpha=node_linewidth)
            if covs is not None:
                circle = plt.Circle(centers[i], covs[i]*1.5, color='steelblue', fill=False, 
                                    alpha=max(class_prob[i]/class_prob.max(), 0.15), linewidth=node_linewidth)
                ax.add_patch(circle)
        else:
            ax.scatter(centers[i][0], centers[i][1], label=f'Node {node}', color='skyblue', s=500, edgecolor='steelblue')
            if covs is not None:
                circle = plt.Circle(centers[i], covs[i]*1.5, color='steelblue', fill=False, alpha=0.15)
                ax.add_patch(circle)
        
        node = i
        ax.text(centers[i][0], centers[i][1], f' {node}', fontsize=13, 
            verticalalignment='center', horizontalalignment='center'
            , fontweight='bold'
            )
        # ---------------------------------------------------------------------------------------


    # shortest path
    for p_i in range(len(path)):
        u_p = path[p_i]
        ax.scatter(centers[u_p][0], centers[u_p][1], s=500, facecolor='none', edgecolor='red', linewidths=3)

        if p_i < len(path)-1:
            v_p = path[p_i+1]
            # plt.plot([centers[u_p][0], centers[v_p][0]], [centers[u_p][1], centers[v_p][1]], color='red')
            # 화살표 추가
            ax.annotate('', xy=[centers[v_p][0], centers[v_p][1]], xytext=[centers[u_p][0], centers[u_p][1]],
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->,head_width=1,head_length=1.5', 
                            lw=2
                            # , headwidth=10, headlength=15
                            )
                )
            mid_x = (centers[u_p][0] + centers[v_p][0]) / 2
            mid_y = (centers[u_p][1] + centers[v_p][1]) / 2

            annot = path_distance[p_i] if len(path_distance) > 0 else adjacent_matrix[u_p][v_p]
            ax.text(mid_x, mid_y, f'{annot:.1f}', fontsize=13,
                    color='darkred', backgroundcolor='none',
                    horizontalalignment='center', verticalalignment='center')

    if point is not None:
        ax.scatter(point[0], point[1], facecolor='red', s=200, marker='*')
        if point_from is not None:
            ax.plot([point_from[0], point[0]], [point_from[1], point[1]], color='red', ls='--')
    ax.set_xlabel('lat')
    ax.set_ylabel('lng')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    if return_plot:
        plt.close()
        return fig
    else:
        plt.show()


# #################################################################################################################
# n_nodes = 50
# random_state = 1
# visualize = False

# # visualize node_map (centers, covs)
# if visualize:
#     node_map = GenerateNodeMap(n_nodes, random_state=random_state)
#     # create node ------------------------------------------------------------------
#     node_map.create_node(node_scale=50, cov_knn=3)

#     # visualize_graph(centers=node_map.centers)
#     visualize_graph(centers=node_map.centers, covs=node_map.covs)


# # visualize node_map with some point
# if visualize:
#     point = rng.uniform(size=2)
#     class_prob, class_argmax = node_map.predict(point)
#     visualize_graph(centers=node_map.centers, covs=node_map.covs, 
#             point=point, class_prob=class_prob, class_argmax=class_argmax)



# # visualize node_map with connection
# if visualize:
#     # create connection ------------------------------------------------------------------
#     node_map.create_connect(connect_scale=0)

#     visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)
#     # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, weight_vis_base_mat=node_map.adj_matrix)


# # visualize node_map with connection and some point
# if example:
#     point = rng.uniform(size=2)
#     class_prob, class_argmax = node_map.predict(point)
#     visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix, covs=node_map.covs, 
#         point=point, class_prob=class_prob, class_argmax=class_argmax)
# #################################################################################################################






# Shortest Path
import heapq

# (Dijkstra Algorithm)
# (Version 4.0 Update)
class Dijkstra:
    def __init__(self, graph=None):
        self.graph = graph
        self.size = len(graph)

    # (Version 4.0 Update)
    def set_graph(self, graph):
        self.graph = graph

    def dijkstra(self, start, end):
        distances = [float('inf')] * self.size
        distances[start] = 0
        priority_queue = [(0, start)]
        visited = [False] * self.size
        previous_nodes = [-1] * self.size

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if visited[current_node]:
                continue
            
            visited[current_node] = True

            for neighbor, weight in enumerate(self.graph[current_node]):
                if weight > 0 and not visited[neighbor]:  # There is a neighbor and it's not visited
                    distance = current_distance + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous_nodes[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        path = self._reconstruct_path(previous_nodes, start, end)
        return distances[end], path

    def _reconstruct_path(self, previous_nodes, start, end):
        path = []
        current_node = end
        while current_node != -1:
            path.append(current_node)
            current_node = previous_nodes[current_node]
        path.reverse()

        if path[0] == start:
            return path
        else:
            return []  # If the path does not start with the start node, return an empty list
    
    def __call__(self, start, end):
        return self.dijkstra(start, end)



# #################################################################################################################
# n_nodes = 50
# random_state = 1


# # (create base graph) 
# node_map = GenerateNodeMap(n_nodes, random_state=random_state)
# node_map.create_node(node_scale=50, cov_knn=3)           # create node
# node_map.create_connect(connect_scale=0)             # create connection


# start_node, end_node = np.random.choice(np.arange(n_nodes), size=2, replace=False)
# dijkstra = Dijkstra(node_map.adj_matrix)
# shortest_distance, path = dijkstra.dijkstra(start_node, end_node)
# feature = np.append(node_map.centers[start_node], node_map.centers[end_node])
# dist = shortest_distance

# #################################################################################################################
# # Generate Graph & shortest-path data-set #######################################################################
# n_nodes = 50
# random_state = 1

# # (create base graph) 
# node_map = GenerateNodeMap(n_nodes, random_state=random_state)
# node_map.create_node(node_scale=50, cov_knn=3)           # create node
# node_map.create_connect(connect_scale=0)             # create connection

# visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix)

# features = []
# dists = []
# for _ in range(1000):
#     start_node, end_node = np.random.choice(np.arange(n_nodes), size=2, replace=False)
#     dijkstra = Dijkstra(node_map.adj_matrix)
#     shortest_distance, path = dijkstra.dijkstra(start_node, end_node)

#     # visualize_graph(centers=node_map.centers, adjacent_matrix=node_map.adj_matrix,
#     #                 path=path, distance=shortest_distance)

#     feature = np.append(node_map.centers[start_node], node_map.centers[end_node])
#     dist = shortest_distance

#     features.append(feature)
#     dists.append(dist)
# #################################################################################################################