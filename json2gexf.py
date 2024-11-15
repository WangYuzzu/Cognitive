import json
import networkx as nx
from pathlib import Path


def convert_to_gephi_format(json_file, output_file):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建NetworkX图
    G = nx.DiGraph()

    # 添加所有实体节点
    entity_nodes = set()  # 用于收集所有实体节点

    # 先从edges中收集所有实体节点
    for edge in data['edges']:
        if edge['type'] == 'relation':  # 只考虑关系边
            entity_nodes.add(edge['source'])
            entity_nodes.add(edge['target'])

    # 添加节点
    for node in entity_nodes:
        G.add_node(node)

    # 只处理关系边
    for edge in data['edges']:
        if edge['type'] == 'relation':  # 只添加关系边
            G.add_edge(
                edge['source'],
                edge['target'],
                type='directed',
                weight=1.0,
                relation_name=edge['relation']
            )

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(G, output_file)

    # 统计信息
    print(f"\nStatistics:")
    print(f"- Nodes: {G.number_of_nodes()}")
    print(f"- Edges: {G.number_of_edges()}")
    print("\nRelation types:")
    relations = set(nx.get_edge_attributes(G, 'relation_name').values())
    for rel in sorted(relations):
        count = sum(1 for _, _, attr in G.edges(data=True) if attr['relation_name'] == rel)
        print(f"- {rel}: {count} edges")


if __name__ == "__main__":
    input_json = "results/machine_learning_kg_graph.json"
    output_gexf = "results/graph.gexf"
    convert_to_gephi_format(input_json, output_gexf)