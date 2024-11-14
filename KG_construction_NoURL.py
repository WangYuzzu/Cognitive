# KG_construction.py
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import json
import networkx as nx
from collections import deque
import asyncio

from fastapi.openapi.models import APIKey
from tqdm import tqdm
from openai import OpenAI
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
        # 存储不需要继续传播的实体
        self.propagation_stopped = set()
        # 存储已经分析过的实体
        self.analyzed_entities = set()

        # API配置
        self.client = OpenAI(api_key=api_key, base_url='https://api.gptapi.us/v1/chat/completions')
        # 用于速率限制
        self.last_request_time = 0
        self.min_request_interval = 1

    def generate_entity_analysis_prompt(self, entities: List[str]) -> str:
        """生成批量分析实体的prompt"""
        return f"""作为AI领域的专家，请分析以下概念：{', '.join(entities)}

        对每个概念进行分析，包括：
        1. 领域关联度分析
        2. 概念粒度分析
        3. 传播必要性分析

        请按如下JSON格式返回：
        {{
            "entity_analysis": [
                {{
                    "entity": "概念名称",
                    "is_ai_related": true/false,
                    "relation_level": "strong/medium/weak",
                    "granularity": "category/subdomain/technique/detail",
                    "should_propagate": true/false,
                    "stop_reason": "如果should_propagate为false，给出原因",
                    "confidence": <0-1的置信度>
                }},
                ...
            ]
        }}"""

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
        """使用openai库调用API"""
        data = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a professional knowledge graph expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()

                # 新版本OpenAI库
                response = self.client.chat.completions.create(**data)
                content = response.choices[0].message.content

                # 或者旧版本
                # response = await openai.ChatCompletion.create(**data)
                # content = response['choices'][0]['message']['content']

                return self._extract_json_from_text(content)

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

        return None

    async def analyze_entities(self, entities: List[str]) -> Dict[str, bool]:
        """批量分析实体，返回是否需要停止传播的字典"""
        # 过滤掉已分析的实体
        entities_to_analyze = [e for e in entities if e not in self.analyzed_entities]
        if not entities_to_analyze:
            return {}

        try:
            # 批量调用API分析实体
            response = await self.call_llm_api(
                self.generate_entity_analysis_prompt(entities_to_analyze)
            )
            data = json.loads(response)
            print(f'批量分析: {data}')

            results = {}
            for analysis in data["entity_analysis"]:
                entity = analysis["entity"]
                self.analyzed_entities.add(entity)

                # 判断是否停止传播
                stop_propagation = (
                        not analysis["is_ai_related"] or
                        (not analysis["should_propagate"] and analysis["confidence"] >= 0.6)
                )

                if stop_propagation:
                    self.propagation_stopped.add(entity)

                results[entity] = stop_propagation

            return results

        except Exception as e:
            print(f"Error analyzing entities: {str(e)}")
            return {}

    async def expand_entity(self, entity: str) -> List[Quad]:
        """扩展单个实体，返回四元组列表"""
        if entity in self.visited_entities:
            return []

        # 如果实体已被标记为停止传播，直接返回
        if entity in self.propagation_stopped:
            return []

        try:
            # 调用LLM API
            response = await self.call_llm_api(self.generate_expansion_prompt(entity))
            data = json.loads(response)

            if not data["is_ai_related"]:
                print(f"Entity {entity} is not AI-related")
                self.propagation_stopped.add(entity)
                return []

            quads = []
            new_entities = set()  # 收集新实体

            for quad_data in data["quads"]:
                quad = Quad(
                    head=quad_data["head"],
                    relation=quad_data["relation"],
                    tail=quad_data["tail"],
                    quad_type=quad_data["type"]
                )
                quads.append(quad)

                if quad.quad_type == "R":
                    self.relation_types.add(quad.relation)
                    new_entities.add(quad.tail)
                else:
                    self.attribute_types.add(quad.relation)

            # 批量分析新实体
            if new_entities:
                propagation_results = await self.analyze_entities(list(new_entities))

                # 将可以传播的实体加入队列
                for entity in new_entities:
                    if not propagation_results.get(entity, False):  # False表示不需要停止传播
                        if entity not in self.visited_entities:
                            self.entity_queue.append(entity)

            self.visited_entities.add(entity)
            return quads

        except Exception as e:
            print(f"Error expanding entity {entity}: {str(e)}")
            return []

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

        print(f"\nPropagation stopped entities ({len(self.propagation_stopped)}):")
        for entity in self.propagation_stopped:
            print(f"- {entity}")

        return self.graph

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
                self.graph.add_edge(
                    quad.head,
                    quad.tail,
                    relation=quad.relation,
                    type="attribute",
                    value=quad.tail
                )

    def save_results(self, output_file_prefix: str = "kg_result"):
        """保存结果"""
        # 保存图谱数据
        graph_file = f"{output_file_prefix}_graph.json"
        self.export_graph(graph_file)

        # 保存统计信息
        stats_file = f"{output_file_prefix}_stats.json"
        analyzer = KnowledgeGraphAnalyzer(self.graph)
        stats = analyzer.get_statistics()

        # 添加传播相关的统计信息
        stats["propagation_stats"] = {
            "total_analyzed_entities": len(self.analyzed_entities),
            "propagation_stopped_count": len(self.propagation_stopped),
            "propagation_stopped_entities": list(self.propagation_stopped)
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 保存实体和关系列表
        details_file = f"{output_file_prefix}_details.txt"
        with open(details_file, 'w', encoding='utf-8') as f:
            f.write("=== Visited Entities ===\n")
            for entity in sorted(self.visited_entities):
                f.write(f"{entity}\n")

            f.write("\n=== Propagation Stopped Entities ===\n")
            for entity in sorted(self.propagation_stopped):
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
                    "value": data.get("value", None)
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "metadata": {
                "num_entities": len(self.graph.nodes()),
                "num_relations": len([e for e in self.graph.edges(data=True) if e[2]["type"] == "relation"]),
                "num_attributes": len([e for e in self.graph.edges(data=True) if e[2]["type"] == "attribute"]),
                "relation_types": list(self.relation_types),
                "attribute_types": list(self.attribute_types),
                "propagation_stopped_entities": list(self.propagation_stopped),
                "analyzed_entities": list(self.analyzed_entities)
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
        api_key = "sk-cjCCsuM7YinkB5dc95BeB9A4C3F443C4B5535bE5EcFdBe4f"
        # My api: sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5
        # Ly api: sk-cjCCsuM7YinkB5dc95BeB9A4C3F443C4B5535bE5EcFdBe4f
        kg_generator = KnowledgeGraphGenerator(api_key)
        print("Starting knowledge graph construction from seed entity: 机器学习")
        graph = await kg_generator.build_knowledge_graph(
            seed_entity="机器学习",
            max_entities=100
        )

        kg_generator.save_results("results/machine_learning_kg_1")

        analyzer = KnowledgeGraphAnalyzer(graph)
        stats = analyzer.get_statistics()
        print("\nKnowledge Graph Statistics:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
