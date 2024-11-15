# KG_construction.py
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import json
import networkx as nx
from collections import deque
import asyncio
from tqdm import tqdm
import aiohttp
import time
import argparse
import os


@dataclass
class Quad:
    """四元组数据结构"""
    head: str
    relation: str
    tail: str
    quad_type: str  # "R" for relation, "A" for attribute


class KnowledgeGraphGenerator:
    def __init__(
            self,
            api_key: str,
            load_from_state: bool = False,
            state_file_prefix: str = 'kg_state'
    ):
        # 初始化变量
        self.graph = nx.MultiDiGraph()
        self.visited_entities = set()
        self.entity_queue = deque()
        self.relation_types = set()
        self.attribute_types = set()
        self.propagation_stopped = set()
        self.analyzed_entities = set()

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

        # 如果需要，从保存的状态加载
        if load_from_state:
            self.load_graph(f'results/{state_file_prefix}_graph.json')
            self.load_state(f'results/{state_file_prefix}_state.json')

    def generate_entity_analysis_prompt(self, entities: List[str]) -> str:
        """生成批量分析实体的prompt"""
        entity_descriptions = []

        # 生成每个实体的描述部分
        for entity in entities:
            entity_descriptions.append(f'"{entity}"：')

        # 将所有实体的描述部分合并
        entities_analysis = "\n".join(entity_descriptions)

        # 完整生成 prompt，加入 few-shot 示例
        return f"""
        作为AI领域的专家，请对以下概念进行批量分析：
        {entities_analysis}

        对于每个概念，请回答以下问题：

        第一部分 - 领域关联度分析：
        1. 这个概念是否属于AI领域？
        2. 如果属于，这个概念与AI领域的关联程度是：
           - 强关联（核心概念）
           - 中度关联（相关概念）
           - 弱关联（边缘概念）

        第二部分 - 概念粒度分析：
        1. 这个概念的粒度级别是：
           - 大类概念（如"机器学习"）
           - 子领域概念（如"监督学习"）
           - 具体技术概念（如"决策树"）
           - 技术细节（如"信息增益计算"）

        第三部分 - 传播必要性分析：
        1. 这个概念是否还需要进一步拆分和传播？
        2. 如果不需要，原因是：
           - 已经是最小粒度的知识点
           - 过于技术细节
           - 不够普遍重要
           - 其他原因（请说明）

        请严格按照以下JSON格式返回，不要包含任何其他内容，以下是回复示例：
        {{
            "entity_analysis": [
                {{
                    "entity": "机器学习",
                    "is_ai_related": true,
                    "relation_level": "strong",
                    "granularity": "category",
                    "should_propagate": true,
                    "stop_reason": "",
                    "confidence": 0.95
                }},
                {{
                    "entity": "随机森林",
                    "is_ai_related": true,
                    "relation_level": "medium",
                    "granularity": "technique",
                    "should_propagate": true,
                    "stop_reason": "",
                    "confidence": 0.88
                }},
                {{
                    "entity": "线性代数",
                    "is_ai_related": true,
                    "relation_level": "weak",
                    "granularity": "category",
                    "should_propagate": false,
                    "stop_reason": "属于边缘概念，不需要进一步传播",
                    "confidence": 0.7
                }},
                {{
                    "entity": "大数据处理",
                    "is_ai_related": true,
                    "relation_level": "strong",
                    "granularity": "category",
                    "should_propagate": true,
                    "stop_reason": "",
                    "confidence": 0.92
                }},
                ...
            ]
        }}
        """

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
        raise ValueError(f"No valid JSON found in response, 对应text: {text}")

    async def call_llm_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """异步调用API"""
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a professional knowledge graph expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
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
                                raise Exception(f"API failed after {max_retries} attempts")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise e

        return None

    async def analyze_entities(self, entities: List[str]) -> Dict[str, bool]:
        """批量分析实体，返回是否需要停止传播的字典"""
        entities_to_analyze = [e for e in entities if e not in self.analyzed_entities]
        if not entities_to_analyze:
            return {}

        try:
            response = await self.call_llm_api(
                self.generate_entity_analysis_prompt(entities_to_analyze)
            )
            data = json.loads(response)

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
                    # 打印"停止传播"信息
                    print(f"停止传播实体: {entity}")
                    print(f"原因: {'不属于AI领域' if not analysis['is_ai_related'] else analysis.get('stop_reason')}")
                    print(f"置信度: {analysis.get('confidence', 'N/A')}")

                results[entity] = stop_propagation

            return results

        except Exception as e:
            print(f"Error analyzing entities: {str(e)}")
            return {}

    async def expand_entity(self, entity: str) -> List[Quad]:
        """扩展单个实体，返回四元组列表"""
        if entity in self.visited_entities:
            print(f'entity: {entity} 已经visited，不进行expand')
            return []

        # 如果实体已被标记为停止传播，直接返回
        if entity in self.propagation_stopped:
            print(f'entity: {entity} 已经是propagated，不进行expand')
            return []

        try:
            # 调用API
            response = await self.call_llm_api(self.generate_expansion_prompt(entity))
            data = json.loads(response)

            should_stop = (not data["is_ai_related"])

            if should_stop:
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
                for ent in new_entities:
                    if not propagation_results.get(ent, False):  # False表示不需要停止传播
                        if ent not in self.visited_entities and ent not in self.entity_queue:
                            self.entity_queue.append(ent)

            self.visited_entities.add(entity)
            return quads

        except Exception as e:
            print(f"Error expanding entity {entity}: {str(e)}")
            return []

    async def build_knowledge_graph(
            self,
            seed_entities: List[str],
            max_entities: int
    ) -> nx.MultiDiGraph:
        """从种子实体开始构建知识图谱"""
        # 直接将种子实体加入队列
        self.entity_queue.extend(seed_entities)

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
                # 检查是否已存在相同的边
                if not self.graph.has_edge(quad.head, quad.tail, key=quad.relation):
                    self.graph.add_edge(
                        quad.head,
                        quad.tail,
                        key=quad.relation,
                        relation=quad.relation,
                        type="relation"
                    )
            else:  # 属性类型
                # 属性作为自循环边，或者使用节点属性
                self.graph.add_edge(
                    quad.head,
                    quad.tail,
                    key=quad.relation,
                    relation=quad.relation,
                    type="attribute",
                    value=quad.tail
                )

    def save_results(self, output_file_prefix: str = "kg_result"):
        """保存结果"""
        # 保存图谱数据
        graph_file = f"{output_file_prefix}_graph.json"
        self.export_graph(graph_file)

        # 保存状态
        state_file = f"{output_file_prefix}_state.json"
        self.save_state(state_file)

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
        print(f"- State: {state_file}")
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
                    "value": data.get("value", None),
                    "key": k
                }
                for u, v, k, data in self.graph.edges(keys=True, data=True)
            ],
            "metadata": {
                "num_entities": len(self.graph.nodes()),
                "num_relations": len([e for e in self.graph.edges(keys=True, data=True) if e[3]["type"] == "relation"]),
                "num_attributes": len([e for e in self.graph.edges(keys=True, data=True) if e[3]["type"] == "attribute"]),
                "relation_types": list(self.relation_types),
                "attribute_types": list(self.attribute_types),
                "propagation_stopped_entities": list(self.propagation_stopped),
                "analyzed_entities": list(self.analyzed_entities)
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

    def save_state(self, state_file: str):
        """保存内部状态"""
        state_data = {
            'visited_entities': list(self.visited_entities),
            'entity_queue': list(self.entity_queue),
            'relation_types': list(self.relation_types),
            'attribute_types': list(self.attribute_types),
            'propagation_stopped': list(self.propagation_stopped),
            'analyzed_entities': list(self.analyzed_entities)
        }
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

    def load_state(self, state_file: str):
        """加载内部状态"""
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        self.visited_entities = set(state_data['visited_entities'])
        self.entity_queue = deque(state_data['entity_queue'])
        self.relation_types = set(state_data['relation_types'])
        self.attribute_types = set(state_data['attribute_types'])
        self.propagation_stopped = set(state_data['propagation_stopped'])
        self.analyzed_entities = set(state_data['analyzed_entities'])

    def load_graph(self, graph_file: str):
        """加载图谱"""
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        self.graph = nx.MultiDiGraph()
        self.graph.add_nodes_from(graph_data['nodes'])
        for edge in graph_data['edges']:
            self.graph.add_edge(
                edge['source'],
                edge['target'],
                key=edge['key'],
                relation=edge['relation'],
                type=edge['type'],
                value=edge.get('value')
            )
        # 更新元数据
        self.relation_types = set(graph_data['metadata']['relation_types'])
        self.attribute_types = set(graph_data['metadata']['attribute_types'])


class KnowledgeGraphAnalyzer:
    """分析知识图谱的工具类"""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        relation_edges = [e for e in self.graph.edges(data=True, keys=True) if e[3]["type"] == "relation"]
        attribute_edges = [e for e in self.graph.edges(data=True, keys=True) if e[3]["type"] == "attribute"]

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
        for _, _, _, data in relation_edges:
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


async def main(args):

    def verify_graph_edges(graph: nx.MultiDiGraph):
        for idx, edge in enumerate(graph.edges(keys=True, data=True)):
            if len(edge) != 4:
                print(f"边 {idx} 的结构不正确: {edge}")
            else:
                u, v, k, data = edge
                # 检查是否包含必要的键
                required_keys = ["relation", "type"]
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f"边 {idx} 缺少键 {missing_keys}: {edge}")

    try:
        kg_generator = KnowledgeGraphGenerator(
            api_key=args.api_key,
            load_from_state=args.load_from_state,
            state_file_prefix=args.state_file_prefix
        )

        seed_entities = args.seed_entities if args.seed_entities else []
        print("开始构建知识图谱")
        await kg_generator.build_knowledge_graph(
            seed_entities=seed_entities,
            max_entities=args.max_entities
        )

        # 验证图中边的结构和数据完整性
        verify_graph_edges(kg_generator.graph)

        kg_generator.save_results(f"results/{args.state_file_prefix}")

        analyzer = KnowledgeGraphAnalyzer(kg_generator.graph)
        stats = analyzer.get_statistics()
        print("\n知识图谱统计信息:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    except Exception as e:
        import traceback
        print(f"错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Generator")
    parser.add_argument('--api_key', required=True, type=str, help='OpenAI API key')
    parser.add_argument('--load_from_state', action='store_true', help='Load from saved state')
    parser.add_argument('--state_file_prefix', type=str, default='kg_state', help='State file prefix')
    parser.add_argument('--seed_entities', default=['机器学习', '自然语言处理', '计算机视觉', '深度学习'], nargs='*',
                        help='List of seed entities')
    parser.add_argument('--max_entities', type=int, default=10, help='Maximum number of entities')

    args = parser.parse_args()

    # 创建结果目录
    os.makedirs('results', exist_ok=True)

    asyncio.run(main(args))
