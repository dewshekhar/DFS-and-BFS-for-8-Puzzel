#Auther Dewashish Pramanik
import numpy as np
from collections import deque
import time


class Node:
    def __init__(self, state, parent, action):
        self.state = state
        self.parent = parent
        self.action = action


class StackFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any((node.state[0] == state[0]).all() for node in self.frontier)
    
    def empty(self):
        return len(self.frontier) == 0
    
    def remove(self):
        if self.empty():
            raise Exception("Empty Frontier")
        else:
            node = self.frontier.pop()
            return node


class Puzzle:
    def __init__(self, start, startIndex, goal, goalIndex):
        self.start = [np.copy(start), startIndex]
        self.goal = [np.copy(goal), goalIndex]
        self.solution_bfs = None
        self.solution_dfs = None

    def neighbors(self, state):
        mat, (row, col) = state
        results = []

        for new_row, new_col in ((row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)):
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                mat1 = np.copy(mat)
                mat1[row][col], mat1[new_row][new_col] = mat1[new_row][new_col], mat1[row][col]
                results.append((f'{mat[row][col]} to {mat[new_row][new_col]}', [mat1, (new_row, new_col)]))

        return results

    def print_solution(self, solution, label):
        if solution is None:
            print(f"No {label} solution found.")
            return

        actions, cells = solution
        print(f"{label.upper()} Steps:", len(actions))
        print("\nStart State:")
        print(self.start[0])
        print("\nGoal State:")
        print(self.goal[0])
        print("\nStates Explored:", self.num_explored)
        print("\nSolution:")
        for action, cell in zip(actions, cells):
            print(f"Action: {action}\n{cell[0]}")
        print("\nGoal Reached!!")

    def solve_bfs(self):
        self.num_explored = 0

        start = Node(state=self.start, parent=None, action=None)
        frontier = deque()
        frontier.append(start)

        explored = set()

        start_time = time.time()

        while frontier:
            node = frontier.popleft()
            self.num_explored += 1

            if (node.state[0] == self.goal[0]).all():
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution_bfs = (actions, cells)
                end_time = time.time()
                self.search_time_bfs = end_time - start_time
                return

            explored.add(tuple(node.state[0].flatten()))

            for action, state in self.neighbors(node.state):
                state_tuple = tuple(state[0].flatten())
                if state_tuple not in explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.append(child)

    def solve_dfs(self):
        self.num_explored = 0

        start = Node(state=self.start, parent=None, action=None)
        frontier = StackFrontier()
        frontier.add(start)

        explored = set()

        start_time = time.time()

        while frontier:
            node = frontier.remove()
            self.num_explored += 1

            if (node.state[0] == self.goal[0]).all():
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution_dfs = (actions, cells)
                end_time = time.time()
                self.search_time_dfs = end_time - start_time
                return

            explored.add(tuple(node.state[0].flatten()))

            for action, state in self.neighbors(node.state):
                state_tuple = tuple(state[0].flatten())
                if state_tuple not in explored:
                    child = Node(state=state, parent=node, action=action)
                    frontier.add(child)


start = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
goal = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])

startIndex = (1, 1)
goalIndex = (1, 0)

puzzle = Puzzle(start, startIndex, goal, goalIndex)

# Solve using BFS
puzzle.solve_bfs()
puzzle.print_solution(puzzle.solution_bfs, "BFS")
print("BFS Search Time:", puzzle.search_time_bfs, "seconds")

# Solve using DFS
puzzle.solve_dfs()
puzzle.print_solution(puzzle.solution_dfs, "DFS")
print("DFS Search Time:", puzzle.search_time_dfs, "seconds")

# Compare DFS and BFS
print("\nComparison of DFS and BFS:")
print("DFS Steps:", len(puzzle.solution_dfs[0]))
print("DFS Search Time:", puzzle.search_time_dfs, "seconds")
print("BFS Steps:", len(puzzle.solution_bfs[0]))
print("BFS Search Time:", puzzle.search_time_bfs, "seconds")

