# stu_score_construction.py

import openai
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime
import time


class OpenAIAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.gptapi.us/v1/chat/completions" # "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # 用于速率限制
        self.last_request_time = 0
        self.min_request_interval = 1  # 最小请求间隔(秒)

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
        # 去除可能的前后缀文本
        text = text.strip()

        # 如果文本已经是一个完整的JSON，直接返回
        if text.startswith('{') and text.endswith('}'):
            return text

        # 寻找第一个 { 和最后一个 } 的位置
        start = text.find('{')
        end = text.rfind('}')

        if start != -1 and end != -1:
            return text[start:end + 1]

        raise ValueError("No valid JSON found in response")

    async def call_api(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """异步调用OpenAI API"""
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a professional Chinese education expert."},
                {"role": "user", "content": prompt}
            ],
            # 增加随机性参数
            "temperature": 0.7,
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
                            # 提取并返回JSON字符串
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


class PromptPool:
    def __init__(self, api_key: str):
        self.api = OpenAIAPI(api_key)

    @staticmethod
    def _validate_student_info(data: Dict[str, Any]) -> bool:
        """验证学生信息是否合法"""
        try:
            assert isinstance(data["age"], int) and 6 <= data["age"] <= 17
            assert isinstance(data["grade"], int) and 1 <= data["grade"] <= 12
            assert isinstance(data["region"], str) and data["region"]
            assert isinstance(data["school_type"], str) and data["school_type"]
            assert isinstance(data["academic_performance"], (int, float)) and 0 <= data["academic_performance"] <= 100
            assert isinstance(data["interests"], list) and len(data["interests"]) > 0
            return True
        except (AssertionError, KeyError):
            return False

    @staticmethod
    def _validate_concept_understanding(data: Dict[str, Any]) -> bool:
        """验证概念理解度数据是否合法"""
        try:
            assert isinstance(data["understanding_level"], (int, float))
            assert 0 <= data["understanding_level"] <= 1
            assert isinstance(data["reasoning"], str) and data["reasoning"]
            return True
        except (AssertionError, KeyError):
            return False

    @staticmethod
    def _validate_negative_sample(data: Dict[str, Any]) -> bool:
        """验证负样本数据是否合法"""
        try:
            assert isinstance(data["understanding_level"], (int, float))
            assert 0 <= data["understanding_level"] <= 1
            assert isinstance(data["reasoning"], str) and data["reasoning"]
            assert isinstance(data["error_type"], str) and data["error_type"]
            assert isinstance(data["hidden_error"], str) and data["hidden_error"]
            # 检查error_type是否为有效值
            valid_error_types = ["忽略关键信息", "过度泛化", "错误关联", "不当推理"]
            assert data["error_type"] in valid_error_types
            return True
        except (AssertionError, KeyError):
            return False

    async def generate_student_info(self) -> Dict[str, Any]:
        """异步生成学生信息"""
        prompt = self.generate_student_basic_info()
        response = await self.api.call_api(prompt)
        if response:
            try:
                data = json.loads(response)
                if not self._validate_student_info(data):
                    raise ValueError("Invalid student info format")
                return data
            except json.JSONDecodeError:
                print("Raw response:", response)
                raise Exception("Failed to parse API response as JSON")
        raise Exception("Failed to get response from API")

    async def get_concept_understanding(
            self,
            student_info: Dict[str, Any],
            concept: str
    ) -> Dict[str, Any]:
        """异步获取概念理解度"""
        prompt = self.generate_concept_understanding(student_info, concept)
        response = await self.api.call_api(prompt)
        if response:
            try:
                data = json.loads(response)
                if not self._validate_concept_understanding(data):
                    raise ValueError("Invalid concept understanding format")
                return data
            except json.JSONDecodeError:
                raise Exception("Failed to parse API response as JSON")
        raise Exception("Failed to get response from API")

    async def get_negative_sample(
            self,
            student_info: Dict[str, Any],
            concept: str,
            original_level: float,
            original_reasoning: str,
    ) -> Dict[str, Any]:
        """异步生成负样本"""
        prompt = self.generate_negative_samples(
            student_info,
            concept,
            original_level,
            original_reasoning,
        )
        response = await self.api.call_api(prompt)
        if response:
            try:
                data = json.loads(response)
                if not self._validate_negative_sample(data):
                    raise ValueError("Invalid negative sample format")
                return data
            except json.JSONDecodeError:
                raise Exception("Failed to parse API response as JSON")
        raise Exception("Failed to get response from API")

    @staticmethod
    def generate_student_basic_info() -> str:
        """生成学生基础信息的prompt（带few-shot示例）"""
        return """你现在是一个中国教育专家，请生成一个合理的中国K12学生信息。参考以下示例，但要生成不同的新数据。

       示例1：
       {
           "age": 12,
           "grade": 6,
           "region": "浙江省",
           "school_type": "普通中学",
           "academic_performance": 82,
           "interests": ["编程", "乐高", "数学"],
           "family_background": {
               "parents_education": ["本科", "硕士"],
               "education_support": "高度重视",
               "learning_resources": "充足"
           },
           "learning_traits": {
               "self_motivation": "强",
               "learning_style": "动手实践",
               "attention_span": "良好"
           },
           "additional_education": {
               "tutoring": ["数学", "编程"],
               "online_courses": ["少儿编程", "趣味科学"],
               "participation": ["机器人竞赛", "数学竞赛"]
           }
       }

       示例2：
       {
           "age": 16,
           "grade": 10,
           "region": "广东省",
           "school_type": "重点中学",
           "academic_performance": 92,
           "interests": ["物理", "人工智能", "篮球"],
           "family_background": {
               "parents_education": ["研究生", "研究生"],
               "education_support": "非常重视",
               "learning_resources": "丰富"
           },
           "learning_traits": {
               "self_motivation": "很强",
               "learning_style": "逻辑思考",
               "attention_span": "持久"
           },
           "additional_education": {
               "tutoring": ["物理", "数学"],
               "online_courses": ["人工智能入门", "高等数学预习"],
               "participation": ["科技创新大赛", "物理竞赛"]
           }
       }

       请严格按照以下规则生成：
       1. 年龄和年级对应关系：
          - 小学：6-11岁，1-6年级
          - 初中：12-14岁，7-9年级
          - 高中：15-17岁，10-12年级
       2. 学习成绩(academic_performance)：
          - 重点学校平均分稍高于普通学校
       3. 家庭背景要合理：
          - 父母教育程度：高中/大专/本科/硕士/博士
          - 教育支持程度要和家庭背景匹配
       4. 学习特征要符合年龄特点
       5. 课外教育要与家庭背景和学生兴趣匹配

       请给出一个新的、合理的学生信息（严格按照JSON格式）：
       """

    @staticmethod
    def generate_concept_understanding(student_info: dict, concept: str) -> str:
        """基于学生信息生成对特定概念的理解程度"""
        return f"""你现在是一个中国教育专家，请基于以下学生信息，预测该学生对给定概念的理解程度。

       学生信息：
       基本信息：
       - 年龄：{student_info['age']}岁
       - 年级：{student_info['grade']}年级
       - 地区：{student_info['region']}
       - 学校类型：{student_info['school_type']}
       - 学习成绩：{student_info['academic_performance']}
       - 兴趣爱好：{', '.join(student_info['interests'])}

       家庭背景：
       - 父母教育程度：{student_info['family_background']['parents_education']}
       - 教育支持度：{student_info['family_background']['education_support']}
       - 学习资源：{student_info['family_background']['learning_resources']}

       学习特征：
       - 自主学习动力：{student_info['learning_traits']['self_motivation']}
       - 学习方式：{student_info['learning_traits']['learning_style']}
       - 注意力水平：{student_info['learning_traits']['attention_span']}

       额外教育：
       - 课外辅导：{', '.join(student_info['additional_education']['tutoring'])}
       - 在线课程：{', '.join(student_info['additional_education']['online_courses'])}
       - 竞赛参与：{', '.join(student_info['additional_education']['participation'])}

       需要评估的概念：{concept}

       请严格按照以下JSON格式返回，不要包含任何其他内容：
       {{
           "understanding_level": <0-1之间的小数>,
           "reasoning": "详细解释为什么这个学生对该概念有这样的理解程度，需要全面考虑学生的各项特征"
       }}

       评估要求：
       1. 认知发展匹配：理解程度要符合学生认知发展阶段
       2. 兴趣相关性：分析兴趣与概念的关联度
       3. 家庭支持：考虑家庭背景对学习的影响
       4. 学习特征：评估学习方式是否有利于理解该概念
       5. 额外教育：分析课外学习对概念理解的促进作用
       6. 教育资源：综合考虑学校和家庭提供的资源
       7. 地区差异：考虑地区教育资源差异的影响
       """

    @staticmethod
    def generate_negative_samples(student_info: dict, concept: str,
                                  original_level: float, original_reasoning: str) -> str:
        """生成负样本的prompt"""
        return f"""你现在是一个教育评估专家，请基于以下信息生成一个负样本（不当的评估）。

        学生信息：
        {student_info}

        概念：{concept}

        正确的评估示例：
        - 理解程度：{original_level}
        - 理由：{original_reasoning}

        请生成一个看似合理但实际存在推理错误的评估。错误类型可以是：
        1. 忽略关键信息
           - 例如：忽略学生的特殊兴趣爱好
           - 例如：忽略学校类型的影响

        2. 过度泛化
           - 例如：仅基于年龄下结论
           - 例如：过分依赖单一特征

        3. 错误关联
           - 例如：将不相关的兴趣误认为有帮助
           - 例如：误解某些特征的影响

        4. 不当推理
           - 例如：使用不恰当的对比
           - 例如：逻辑链条存在跳跃

        请按以下JSON格式返回：
        {{
            "understanding_level": <理解程度值>,
            "reasoning": "包含推理错误的理由",
            "error_type": "错误的类型（上述四种之一）",
            "hidden_error": "解释该评估中隐藏的具体错误"
        }}

        要求：
        1. 错误要隐晦，不要太明显
        2. 评估要看起来专业和合理
        3. 理由要有一定说服力
        4. 避免明显违背教育常识
        """

    @staticmethod
    def batch_generate_prompt(num_samples: int) -> str:
        """批量生成数据的prompt"""
        return f"""请生成{num_samples}个不同的样本，每个样本包含学生信息和概念理解评估。
        格式要求同上，返回一个样本列表。每个样本都要足够独特且合理。
        """


async def main():
    # 使用示例
    try:
        # 初始化PromptPool（需要替换为你的API密钥）
        api_key = "sk-bCY6IwFu0nbiaRTF528e01387bFf4c3eB2E224B6201129A5"
        pool = PromptPool(api_key)

        num_sample = 5  # 采样数
        for _ in range(num_sample):
            # 1. 生成学生信息
            print("Generating student info...")
            student_info = await pool.generate_student_info()
            print("Student info:", json.dumps(student_info, ensure_ascii=False, indent=2))

            # 2. 生成概念理解
            concept = "机器学习"
            print(f"\nGenerating concept understanding for '{concept}'...")
            understanding = await pool.get_concept_understanding(student_info, concept)
            print("Understanding:", json.dumps(understanding, ensure_ascii=False, indent=2))

            # 3. 生成负样本
            print("\nGenerating negative sample...")
            negative_sample = await pool.get_negative_sample(
                student_info=student_info,
                concept=concept,
                original_level=understanding["understanding_level"],
                original_reasoning=understanding["reasoning"],
            )
            print("Negative sample:", json.dumps(negative_sample, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
