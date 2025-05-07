class Node:
    def __init__(self, name):
        self.name = name
        self.edges = []  # list of (next_node, target_angle)

    def add_edge(self, next_node, target_angle):
        self.edges.append((next_node, target_angle))

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, name):
        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def add_edge(self, from_node_name, to_node_name, target_angle, bidirectional=True):
        from_node = self.add_node(from_node_name)
        to_node = self.add_node(to_node_name)
        from_node.add_edge(to_node, target_angle)
        if bidirectional:
            reverse_angle = self.normalize_angle(target_angle + 180)
            to_node.add_edge(from_node, reverse_angle)

    def normalize_angle(self, angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def angle_to_code(self, delta_angle):
        delta = self.normalize_angle(delta_angle)
        if abs(delta) < 20:
            return 2  # 直行
        elif abs(delta - 180) < 20 or abs(delta + 180) < 20:
            return 3  # 掉头
        elif delta > 0:
            return 1  # 右转
        else:
            return 0  # 左转

    def find_path_with_codes(self, start, goal, start_angle):
        path = []
        codes = []

        def dfs(current_node, visited, current_angle):
            if current_node.name == goal:
                return True, current_angle
            visited.add(current_node.name)

            for next_node, target_angle in current_node.edges:
                if next_node.name not in visited:
                    delta = self.normalize_angle(target_angle - current_angle)
                    new_angle = target_angle

                    path.append(next_node.name)
                    codes.append(self.angle_to_code(delta))

                    found, final_angle = dfs(next_node, visited, new_angle)
                    if found:
                        return True, final_angle

                    path.pop()
                    codes.pop()
            return False, current_angle

        start_node = self.nodes.get(start)
        if not start_node:
            return None, None

        path.append(start)
        found, _ = dfs(start_node, set(), start_angle)
        if found:
            return path, codes
        else:
            return None, None

def build_car_track_graph():
    g = Graph()
    g.add_edge('S', 'A1_L', -90)
    g.add_edge('S', 'A1_S', 0)
    g.add_edge('S', 'A1_R', 90)

    g.add_edge('A1_L', 'A2_LL', -180)
    g.add_edge('A1_L', 'A2_LR', 0)
    g.add_edge('A1_S', 'A2_SL', -90)
    g.add_edge('A1_S', 'A2_SR', 90)
    g.add_edge('A1_R', 'A2_RL', 0)
    g.add_edge('A1_R', 'A2_RR', 180)

    g.add_edge('A2_LL', 'E1', -180)
    g.add_edge('A2_LL', 'E2', -90)
    g.add_edge('A2_LR', 'E3', -90)
    g.add_edge('A2_LR', 'E4', 0)
    g.add_edge('A2_SL', 'E5', 0)
    g.add_edge('A2_SR', 'E6', 0)
    g.add_edge('A2_RL', 'E7', 0)
    g.add_edge('A2_RL', 'E8', 90)
    g.add_edge('A2_RR', 'E9', 90)
    g.add_edge('A2_RR', 'E10', 180)
    return g

def plan_route(start, goals, start_angle=0):
    """
    输入起点 start (str)、目标列表 goals (list of str)、初始角度 start_angle (默认0度)
    输出整体路径和动作编码（0左转，1右转，2直行，3掉头）
    """
    g = build_car_track_graph()
    current = start
    current_angle = start_angle
    full_path = []
    full_codes = []

    for target in goals:
        path, codes = g.find_path_with_codes(current, target, current_angle)
        if path:
            full_path += path[1:]  # 第一个点是current，跳过
            full_codes += codes
            current = target
            if path:
                # 更新当前角度为最后一跳的角度
                last_edge = (path[-2], path[-1])
                for next_node, target_angle in g.nodes[last_edge[0]].edges:
                    if next_node.name == last_edge[1]:
                        current_angle = target_angle
                        break
        else:
            print(f"从 {current} 无法到达 {target}")
            break

    # 打印路线和数字串
    print("完整路径:", ' -> '.join([start] + full_path))
    print("动作数字串:", '-'.join(map(str, full_codes)))

    return full_path, full_codes

# 测试
if __name__ == "__main__":
    # 单目标测试
    print("【单目标】从S到E9")
    plan_route('A2_RL', ['E7'], start_angle=90)  # 测试不同的起始角度

    print("\n【多目标】从S出发，连续前往 E1、E7")
    plan_route('S', ['E1','E7'], start_angle=90)  # 测试不同的起始角度
