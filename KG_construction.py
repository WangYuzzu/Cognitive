# knowledge_graph_generator.py
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import json
import networkx as nx
from collections import deque
import asyncio
from tqdm import tqdm
import aiohttp
import time
from datetime import datetime


@dataclass
class Quad:
    """四元组数据结构"""
    head: str
    relation: str
    tail: str
    quad_type: str  # "R" for relation, "A" for attribute


class KnowledgeGraphGenerator:
    def __init__(self, api_key: str):
        # 使用NetworkX存储图谱
        self.graph = nx.MultiDiGraph()
        # 存储已访问的实体
        self.visited_entities = set()
        # 待访问的实体队列
        self.entity_queue = deque()
        # 记录所有关系类型
        self.relation_types = set()
        # 记录所有属性类型
        self.attribute_types = set()

        # API配置
        self.api_key = api_key
        self.base_url = "https://api.gptapi.us/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # 用于速率限制
        self.last_request_time = 0
        self.min_request_interval = 1

    def generate_expansion_prompt(self, entity: str) -> str:
        """生成用于实体扩展的prompt"""
        return f"""你是一名专业的知识图谱从业者，针对关键词实体: {entity}，请你：

        1. 首先判断这个词是否与人工智能领域存在强关联，如果不存在，则输出："不属于人工智能领域"

        2. 如果属于人工智能领域，请生成一系列四元组：
           - 格式为<实体, 关系, 实体, R>或<实体, 属性, 属性值, A>
           - 主实体必须是"{entity}"
           - 关联实体必须与AI领域相关
           - 关系要表达实体间的语义联系
           - 属性要表达实体的内在特性

        请严格按照以下JSON格式返回，不要包含任何其他内容：
        {{
            "is_ai_related": true/false,
            "quads": [
                {{
                    "head": "{entity}",
                    "relation": "关系名称",
                    "tail": "关联实体/属性值",
                    "type": "R/A"
                }},
                ...
            ]
        }}

        要求：
        1. 关系要准确具体
        2. 关联实体要在合适的粒度级别
        3. 每个实体生成8-12个四元组
        4. 优先生成与其他AI概念的关系
        5. 关系描述要简洁，通常2-4个字
        6. 实体名称要规范，避免同义词
        """

    async def _wait_for_rate_limit(self):
        """实现简单的速率限制"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    @staticmethod
    def _extract_json_from_text(text: str) -> str:
        """从文本中提取JSON字符串"""
        text = text.strip()
        if text.startswith('{') and text.endswith('}'):
            return text
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end + 1]
        raise ValueError("No valid JSON found in response")

    async def call_llm_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """异步调用OpenAI API"""
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a professional knowledge graph expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            self.base_url,
                            headers=self.headers,
                            json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content']
                            return self._extract_json_from_text(content)
                        else:
                            error_detail = await response.text()
                            print(f"API Error {response.status}: {error_detail}")
                            if attempt == max_retries - 1:
                                raise Exception(f"OpenAI API failed after {max_retries} attempts")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

        return None

    async def expand_entity(self, entity: str) -> List[Quad]:
        """扩展单个实体，返回四元组列表"""
        if entity in self.visited_entities:
            return []

        try:
            # 调用LLM API
            response = await self.call_llm_api(self.generate_expansion_prompt(entity))
            data = json.loads(response)

            if not data["is_ai_related"]:
                print(f"Entity {entity} is not AI-related")
                return []

            quads = []
            for quad_data in data["quads"]:
                quad = Quad(
                    head=quad_data["head"],
                    relation=quad_data["relation"],
                    tail=quad_data["tail"],
                    quad_type=quad_data["type"]
                )
                quads.append(quad)

                # 记录关系和属性类型
                if quad.quad_type == "R":
                    self.relation_types.add(quad.relation)
                    # 将新实体加入队列
                    if quad.tail not in self.visited_entities:
                        self.entity_queue.append(quad.tail)
                else:
                    self.attribute_types.add(quad.relation)

            self.visited_entities.add(entity)
            return quads

        except Exception as e:
            print(f"Error expanding entity {entity}: {str(e)}")
            return []

    def add_quads_to_graph(self, quads: List[Quad]):
        """将四元组添加到图中"""
        for quad in quads:
            if quad.quad_type == "R":  # 关系类型
                self.graph.add_edge(
                    quad.head,
                    quad.tail,
                    relation=quad.relation,
                    type="relation"
                )
            else:  # 属性类型
                # 直接添加属性值作为边的属性，而不是创建新节点
                self.graph.add_edge(
                    quad.head,
                    quad.tail,  # 直接使用属性值作为目标节点
                    relation=quad.relation,
                    type="attribute",
                    value=quad.tail
                )

    async def build_knowledge_graph(self, seed_entity: str, max_entities: int = 10) -> nx.MultiDiGraph:
        """从种子实体开始构建知识图谱"""
        self.entity_queue.append(seed_entity)

        with tqdm(total=max_entities) as pbar:
            while self.entity_queue and len(self.visited_entities) < max_entities:
                current_entity = self.entity_queue.popleft()
                print(f"\nProcessing entity: {current_entity}")

                quads = await self.expand_entity(current_entity)
                if quads:
                    print(f"Found {len(quads)} quads:")
                    for quad in quads:
                        print(f"- {quad}")

                    self.add_quads_to_graph(quads)
                    pbar.update(1)
                else:
                    print(f"No quads found for entity: {current_entity}")

        print(f"\nVisited entities ({len(self.visited_entities)}):")
        for entity in self.visited_entities:
            print(f"- {entity}")

        return self.graph

    def save_results(self, output_file_prefix: str = "kg_result"):
        """保存结果"""
        # 保存图谱数据
        graph_file = f"{output_file_prefix}_graph.json"
        self.export_graph(graph_file)

        # 保存统计信息
        stats_file = f"{output_file_prefix}_stats.json"
        analyzer = KnowledgeGraphAnalyzer(self.graph)
        stats = analyzer.get_statistics()

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 保存实体和关系列表
        details_file = f"{output_file_prefix}_details.txt"
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write("=== Visited Entities ===\n")
            for entity in sorted(self.visited_entities):
                f.write(f"{entity}\n")

            f.write("\n=== Relation Types ===\n")
            for relation in sorted(self.relation_types):
                f.write(f"{relation}\n")

            f.write("\n=== Attribute Types ===\n")
            for attribute in sorted(self.attribute_types):
                f.write(f"{attribute}\n")

        print(f"\nResults saved:")
        print(f"- Graph: {graph_file}")
        print(f"- Stats: {stats_file}")
        print(f"- Details: {details_file}")

    def export_graph(self, output_file: str):
        """导出知识图谱为JSON格式"""
        graph_data = {
            "nodes": list(self.graph.nodes()),
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "relation": data["relation"],
                    "type": data["type"],
                    "value": data.get("value", None)  # 对于属性边
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "metadata": {
                "num_entities": len(self.graph.nodes()),
                "num_relations": len([e for e in self.graph.edges(data=True) if e[2]["type"] == "relation"]),
                "num_attributes": len([e for e in self.graph.edges(data=True) if e[2]["type"] == "attribute"]),
                "relation_types": list(self.relation_types),
                "attribute_types": list(self.attribute_types)
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)


class KnowledgeGraphAnalyzer:
    """分析知识图谱的工具类"""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        relation_edges = [e for e in self.graph.edges(data=True) if e[2]["type"] == "relation"]
        attribute_edges = [e for e in self.graph.edges(data=True) if e[2]["type"] == "attribute"]

        return {
            "total_entities": len(self.graph.nodes()),
            "total_relations": len(relation_edges),
            "total_attributes": len(attribute_edges),
            "relation_distribution": self._get_relation_distribution(relation_edges),
            "central_entities": self._get_central_entities(),
            "leaf_entities": self._get_leaf_entities()
        }

    def _get_relation_distribution(self, relation_edges) -> Dict:
        """获取关系类型分布"""
        distribution = {}
        for _, _, data in relation_edges:
            relation = data["relation"]
            distribution[relation] = distribution.get(relation, 0) + 1
        return distribution

    def _get_central_entities(self, top_k: int = 10) -> List[tuple]:
        """获取中心实体（基于度中心性）"""
        degree_centrality = nx.degree_centrality(self.graph)
        return sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _get_leaf_entities(self) -> List[str]:
        """获取叶子实体（度为1的节点）"""
        return [node for node in self.graph.nodes() if self.graph.degree(node) == 1]

    def find_path(self, source: str, target: str, cutoff: Optional[int] = None) -> List[List[str]]:
        """找到两个实体之间的所有路径"""
        try:
            return list(nx.all_simple_paths(self.graph, source, target, cutoff=cutoff))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []


async def main():
    try:
        # 初始化生成器（替换为你的API密钥）
        api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
        kg_generator = KnowledgeGraphGenerator(api_key)

        # 从"机器学习"开始构建图谱，限制为10个实体
        print("Starting knowledge graph construction from seed entity: 机器学习")
        graph = await kg_generator.build_knowledge_graph(
            seed_entity="机器学习",
            max_entities=100
        )

        # 保存结果
        kg_generator.save_results("results/machine_learning_kg")

        # 显示统计信息
        analyzer = KnowledgeGraphAnalyzer(graph)
        stats = analyzer.get_statistics()
        print("\nKnowledge Graph Statistics:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())