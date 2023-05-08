import numpy as np
from collections import deque

### [ Sort ] #############################################################
arr = np.random.randint(500,size=100)
np.array(merge_sort(list(arr)))

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    left = merge_sort(left)
    right = merge_sort(right)
    
    return merge(left, right)

def merge(left, right):
    result = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
            
    result += left[i:]
    result += right[j:]
    
    return result



# quick_sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = []
        right = []
        for i in range(1, len(arr)):
            if arr[i] < pivot:
                left.append(arr[i])
            else:
                right.append(arr[i])
        return np.array(list(quick_sort(left)) + [pivot] + list(quick_sort(right)))

# bubble_sort
def bubble_sort(alist):
    for i in range(len(alist))[::-1]:
        for j in range(i):
            if alist[j] > alist[j+1]:
                temp = alist[j]
                alist[j] = alist[j+1]
                alist[j+1] = temp
    return alist


# selection_sort
def selection_sort(alist):
    for i in range(len(alist)-1):
        j = i+1
        min_idx = j

        while j < len(alist):
            if alist[j] < alist[min_idx]:
                min_idx = j
            j += 1
        
        temp = alist[i]
        alist[i] = alist[min_idx]
        alist[min_idx] = temp
    return alist



# insert_sort
def insert_sort(alist):
    for i in range(len(alist)-1):
        j = i
        while alist[j] > alist[j+1]:
            temp = alist[j]
            alist[j] = alist[j+1]
            alist[j+1] = temp
            j -= 1
    return alist


################################################################


### [ Search ] #############################################################
arr = sorted(np.random.randint(500,size=100))
binary_search(arr, 49)


# binary_search
def binary_search(alist, item):
    first = 0
    last = len(alist)-1

    found = False

    while (first < last) and (not found):
        mid_point = int((first + last)//2)

        if alist[mid_point] == item:
            found = True
        else:
            if item < alist[mid_point]:
                last = mid_point - 1
            else:
                first = mid_point + 1
    return mid_point




### [ Graph ] #############################################################

graph = [
    []
    ,[2,3,8]
    ,[1,7]
    ,[1,4,5]
    ,[3,5]
    ,[3,4]
    ,[7]
    ,[2,6,8]
    ,[1,7]
    ]
visited = [False]*len(graph)


bfs(graph, 1, visited)
dfs(graph, 1, visited)
dfs_self(graph, 1, visited)

# BFS
def bfs(graph, node, visited):
    queue = deque([node])
    visited[node] = True

    while queue:
        cur = queue.popleft()
        print(cur, end=' ')
        
        for i in graph[cur]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

# DFS
    def dfs(graph, node, visited):
        stack = [node]
        visited[node] = True

        while stack:
            cur = stack.pop()
            print(cur, end=' ')

            for i in graph[cur]:
                if not visited[i]:
                    stack.append(i)
                    visited[i] = True
    
# DFS: 재귀함수
    def dfs(graph, node, visited):
        visited[node] = True
        print(node, end=' ')

        for v in graph[node]:
            if visited[v] is not True:
                dfs_self(graph, v, visited)

################################################################




# union-find
### [ Path Graph ] #############################################################
uv_graph =[
 [7,8]
,[1,2]
,[1,5]
,[2,3]
,[2,6]
,[3,4]
,[4,7]
,[5,6]
,[6,4]
]

uv_graph =[
 [1,2]
,[2,3]
,[3,1]
,[4,2]
,[4,5]
,[5,7]
,[7,6]
,[6,5]
,[8,5]
,[11,3]
,[11,8]
,[8,9]
,[9,10]
,[10,11]
]




n_nodes = max(map(max, uv_graph))+1

indegree = [0] * n_nodes    # 진입차수
graph = [[] for _ in range(n_nodes)]
for u, v in uv_graph:
    graph[u].append(v)
    indegree[v] += 1


def topology_sort(graph, indegree):
    result = []
    queue = deque()

    for i in range(1, len(indegree)):
        if indegree[i] == 0:
            queue.append(i)

    while queue:
        cur = queue.popleft()
        result.append(cur)
        print(cur, end=' ')
        
        for i in graph[cur]:
            indegree[i] -= 1

            if indegree[i] == 0:
                queue.append(i)
################################################################













































import numpy as np

class Node:
    def __init__(self, val):
        self.val = val
        self.leftChild = None
        self.rightChild = None
    
    def get(self):
        return self.val
    
    def set(self, val):
        self.val = val
        
    def getChildren(self):
        children = []
        if(self.leftChild != None):
            children.append(self.leftChild)
        if(self.rightChild != None):
            children.append(self.rightChild)
        return children
        
class BST:
    def __init__(self):
        self.root = None

    def setRoot(self, val):
        self.root = Node(val)

    def insert(self, val):
        if(self.root is None):
            self.setRoot(val)
        else:
            self.insertNode(self.root, val)

    def insertNode(self, currentNode, val):
        if(val <= currentNode.val):
            if(currentNode.leftChild):
                self.insertNode(currentNode.leftChild, val)
            else:
                currentNode.leftChild = Node(val)
        elif(val > currentNode.val):
            if(currentNode.rightChild):
                self.insertNode(currentNode.rightChild, val)
            else:
                currentNode.rightChild = Node(val)

    def find(self, val):
        return self.findNode(self.root, val)

    def findNode(self, currentNode, val):
        if(currentNode is None):
            return False
        elif(val == currentNode.val):
            return True
        elif(val < currentNode.val):
            return self.findNode(currentNode.leftChild, val)
        else:
            return self.findNode(currentNode.rightChild, val)
    
    def traverse(self):
        return self.traverseNode(self.root)
    
    def traverseNode(self, currentNode):
        result = []
        if (currentNode.leftChild is not None):
            result.extend(self.traverseNode(currentNode.leftChild))
        if (currentNode is not None):
            result.extend([currentNode.val])
        if (currentNode.rightChild is not None):
            result.extend(self.traverseNode(currentNode.rightChild))
        return result

import numpy as np
rst = np.random.RandomState(1)
arr = np.arange(30)
arr_t = rst.permutation(arr)[:10]
arr_t

bst = BST()

for v in arr_t:
    bst.insert(v)

bst.root.val
bst.root.leftChild.val
bst.root.rightChild.val
bst.traverseNode(bst.root)





# import math
import heapq

n_node = 7
dist_mat = np.ones((n_node, n_node)) * np.inf
diag_i = range(dist_mat.shape[0])
dist_mat[diag_i, diag_i] = 0
# dist_mat

graph_input = [
 [1, 2, 2]
,[1, 4, 1]
,[1, 3, 5]
,[2, 1, 2]
,[2, 4, 2]
,[2, 3, 3]
,[4, 1, 1]
,[4, 2, 2]
,[4, 3, 3]
,[4, 5, 1]
,[3, 1, 5]
,[3, 2, 3]
,[3, 4, 3]
,[3, 5, 1]
,[3, 6, 5]
,[5, 4, 1]
,[5, 3, 1]
,[5, 6, 2]
,[6, 3, 5]
,[6, 5, 2]
]
len(graph_input)
for node_input in graph_input:
    i, j, w = node_input
    dist_mat[i, j] = w


# g = Graph(np.arange(1,7).astype('str'))
# for node_input in graph_input:
#     i, j, w = node_input
#     g.add_edge(str(i), str(j), w)
# print( shortest_path('1','3') )



# case_iter = int(input())
# case_iter = 1
# for _ in range(case_iter):
n_node = 4
graph = [[] for _ in range(4)]



for node_input in graph_input:
    u, v, c = node_input
    graph[u].append((c, v))


hq = []
alarm = [True] + [False]*(n-1)     # Visited, Check
heapq.heappush(hq, (0,0))       # 정점 초기화
node_dist = [0] + [-1 for _ in range(n-1)]     # 초기화 Inf

dist, node = heapq.heappop(hq)  # 최상위노드

# while sum(alarm) < n:
while not alarm[n-1]:
    if alarm[node]:
        for d, v in graph[node]:
            if not alarm[v]:
                new_dist = d + node_dist[node]
                if node_dist[v] == -1 or new_dist < node_dist[v]:
                    node_dist[v] = new_dist
                    heapq.heappush(hq, (node_dist[v], v))

        dist, node = heapq.heappop(hq)
        alarm[node] = True
    else:
        dist, node = heapq.heappop(hq)
    
    if not hq:
        break

if alarm[n-1]:
    result = node_dist[n-1]
else:
    result = -1
    
print(result)








q1 = [(3,8), (2,9),(1,10)]

heapify(q1)
heappop(q1)


# dijkstra implementation from MIT 6006 course lesson #16
from collections import defaultdict
import math
from heapq import heapify, heappush, heappop
import networkx as nx

# utility: priority queue
class Pq:
    def __init__(self):
        self.queue = []
        
    def __str__(self):
        return str(self.queue)
        
    def insert(self, item):
        heappush(self.queue, item)
    
    def extract_min(self):
        return heappop(self.queue)[1]
    
    def update_priority(self, key, priority):
        for v in self.queue:
            if v[1] == key:
                v[0] = priority
        heapify(self.queue)
    
    def empty(self):
        return len(self.queue) == 0

# utility: Graph
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(lambda: [])
    
    def add_edge(self, v, u, w):
        self.graph[v].append((u, w))
        
    def __str__(self):
        result = ''
        for v in self.V:
            result += f'{v}: {str(self.graph[v])}, \n'
        return result
        
def dijkstra(graph, s):
    Q = Pq() # priority queue of vertices
    		 # [ [distance, vertex], ... ] 
    d = dict.fromkeys(graph.V, math.inf) # distance pair 
                                         # will have default value of Infinity
    pi = dict.fromkeys(graph.V, None) # map of parent vertex
    								  # useful for finding shortest path	
    
    # initialize
    d[s] = 0
    
    # update priority if prior path has larger distance
    def relax(u, v, w):
        if d[v] > d[u] + w:
            d[v] = d[u] + w
            Q.update_priority(v, d[v])
            pi[v] = u
    
    # initialize queue
    for v in graph.V:
        Q.insert([d[v], v])
    
    while not Q.empty():
        u = Q.extract_min()
        for v, w in graph.graph[u]:
            relax(u, v, w)
        
    return d, pi

def shortest_path(s, t):
    d, pi = dijkstra(g, s)
    path = [t]
    current = t
    
    # if parent pointer is None,
    # then it's the source vertex
    while pi[current]:
        path.insert(0, pi[current])
        # set current to parent
        current = pi[current]
        
    if s not in path:
        return f'unable to find shortest path staring from "{s}" to "{t}"'
    
    return f'{" > ".join(path)}'

g = Graph(['A', 'B', 'C', 'D', 'E'])


g.add_edge('A', 'B', 10)
g.add_edge('A', 'C', 3)
g.add_edge('B', 'C', 1)
g.add_edge('C', 'B', 4)
g.add_edge('B', 'D', 2)
g.add_edge('C', 'D', 8)
g.add_edge('D', 'E', 7)
g.add_edge('E', 'D', 9)
g.add_edge('C', 'E', 2)

print( shortest_path('B', 'E') )

G = nx.DiGraph()
G.add_weighted_edges_from([\
    ('A', 'B', 10), ('A', 'C', 3), ('B', 'C', 1), ('C', 'B', 4), \
    ('B', 'D', 2), ('C', 'D', 8), ('D', 'E', 7), ('E', 'D', 9), ('C', 'E', 2)])
nx.draw(G, with_labels = True, node_color='b', font_color='w')

import numpy as np
np.log2(26)















































